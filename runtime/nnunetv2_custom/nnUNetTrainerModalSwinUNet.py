import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from scipy.ndimage import distance_transform_edt as distance_transform_edt

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.modal_swin_unet import (
    ModalAttnSwinUNet,
    ParallelModalSwinUNetConcatFusion,
    ParallelModalUNetNoSwin,
    ParallelModalSwinUNet,
    ParallelSwinUNetNoMAF,
    ModalSwinUNet,
    SearchedModalAttnSwinUNet,
    SearchedModalSwinUNet,
    SearchedParallelModalSwinUNet,
    initialize_he,
)


def _base_channels_from_plan_kwargs(arch_init_kwargs):
    features = arch_init_kwargs.get("features_per_stage", None)
    if isinstance(features, (list, tuple)) and len(features) > 0:
        base_channels = int(features[0])
    else:
        base_channels = 32
    return max(16, min(base_channels, 32))


def _searched_arch_path():
    path = os.environ.get("NNUNET_SEARCH_ARCH", "") or os.environ.get("NNUNETV2_SEARCH_ARCH", "")
    if not path:
        raise RuntimeError(
            "Searched nnU-Net trainer requires NNUNET_SEARCH_ARCH=/path/to/best_arch_GLOBAL.json"
        )
    return path


def _num_epochs(default):
    return int(os.environ.get("NNUNET_NUM_EPOCHS", default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


class SeededTrainerMixin:
    """Set repeatable initialization seeds while retaining nnU-Net's pipeline."""

    def _set_run_seed(self):
        run_seed = int(os.environ.get("NNUNET_RUN_SEED", "42"))
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

    def initialize(self):
        self._set_run_seed()
        return super().initialize()

    def get_dataloaders(self):
        self._set_run_seed()
        return super().get_dataloaders()


def _build_network(
        network_cls,
        arch_init_kwargs,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
        base_channels=None,
        **extra_kwargs,
):
    if base_channels is None:
        base_channels = _base_channels_from_plan_kwargs(arch_init_kwargs)
    network = network_cls(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        base_channels=int(base_channels),
        deep_supervision=enable_deep_supervision,
        swin_window_size=(3, 3, 3),
        **extra_kwargs,
    )
    network.apply(initialize_he)
    return network


class NNUNetRegionBoundaryTCFPLoss(nn.Module):
    """
    Region-wise boundary/curvature-aware Tversky-BCE loss with a TC distance-aware
    false-positive penalty, adapted to nnU-Net's region-format deep supervision.
    """

    def __init__(
            self,
            boundary_weight=0.5,
            curvature_weight=2.0,
            et_weight=1.5,
            bce_weight=0.5,
            tc_fp_weight=0.10,
            tc_fp_distance=32.0,
            tc_fp_power=1.0,
            tc_fp_min_distance=0.0,
            smooth=1e-5,
    ):
        super().__init__()
        self.boundary_weight = float(boundary_weight)
        self.curvature_weight = float(curvature_weight)
        self.et_weight = float(et_weight)
        self.bce_weight = float(bce_weight)
        self.tc_fp_weight = float(tc_fp_weight)
        self.tc_fp_distance = float(tc_fp_distance)
        self.tc_fp_power = float(tc_fp_power)
        self.tc_fp_min_distance = float(tc_fp_min_distance)
        self.smooth = float(smooth)
        kernel = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)
        self.register_buffer("laplacian_kernel", kernel)

    @staticmethod
    def _as_region_target(target):
        if isinstance(target, (list, tuple)):
            target = target[0]
        if target.ndim == 5 and target.shape[1] == 3:
            return target.float()
        if target.ndim == 5 and target.shape[1] == 1:
            label = target[:, 0].long()
        elif target.ndim == 4:
            label = target.long()
        else:
            raise ValueError(f"Unsupported nnU-Net target shape for proposed loss: {tuple(target.shape)}")

        wt = (label > 0).float().unsqueeze(1)
        tc = ((label == 1) | (label == 3)).float().unsqueeze(1)
        et = (label == 3).float().unsqueeze(1)
        return torch.cat([wt, tc, et], dim=1)

    def _match_target_size(self, target, spatial_size):
        if tuple(target.shape[2:]) == tuple(spatial_size):
            return target
        return F.interpolate(target.float(), size=spatial_size, mode="nearest")

    def _target_weight_map(self, gt):
        if self.boundary_weight <= 0.0 and self.curvature_weight <= 0.0:
            return torch.ones_like(gt)

        with torch.no_grad():
            gt_bin = (gt > 0.5).float()
            weight = torch.ones_like(gt_bin)
            if self.boundary_weight > 0.0:
                dilated = F.max_pool3d(gt_bin, kernel_size=3, stride=1, padding=1)
                eroded = -F.max_pool3d(-gt_bin, kernel_size=3, stride=1, padding=1)
                boundary = (dilated - eroded).clamp_(0.0, 1.0)
                weight = weight + self.boundary_weight * boundary
            if self.curvature_weight > 0.0:
                curv = torch.abs(F.conv3d(gt_bin, self.laplacian_kernel.to(gt_bin), padding=1))
                max_per_sample = curv.amax(dim=tuple(range(1, curv.ndim)), keepdim=True).clamp_min_(1e-6)
                curv = curv / max_per_sample
                weight = weight + self.curvature_weight * curv
        return weight

    def _region_loss(self, logits, gt, alpha, beta):
        gt = gt.to(device=logits.device, dtype=logits.dtype)
        weight = self._target_weight_map(gt).to(dtype=logits.dtype)
        probs = torch.sigmoid(logits)
        tp = (probs * gt * weight).sum()
        fp = (probs * (1.0 - gt) * weight).sum()
        fn = ((1.0 - probs) * gt * weight).sum()
        tversky = (tp + self.smooth) / (tp + alpha * fp + beta * fn + self.smooth)
        bce = F.binary_cross_entropy_with_logits(logits, gt, weight=weight)
        return (1.0 - tversky) + self.bce_weight * bce

    def _tc_fp_loss(self, tc_logits, gt_tc):
        if self.tc_fp_weight <= 0.0:
            return tc_logits.sum() * 0.0
        gt_np = (gt_tc.detach().cpu().numpy() > 0.5).astype(np.uint8)
        maps = []
        for mask in gt_np[:, 0]:
            if mask.any():
                dist = distance_transform_edt(1 - mask).astype(np.float32)
            else:
                dist = np.full(mask.shape, self.tc_fp_distance, dtype=np.float32)
            if self.tc_fp_min_distance > 0.0:
                denom = max(self.tc_fp_distance - self.tc_fp_min_distance, self.smooth)
                dist = (dist - self.tc_fp_min_distance) / denom
            else:
                dist = dist / max(self.tc_fp_distance, self.smooth)
            dist = np.clip(dist, 0.0, 1.0)
            if self.tc_fp_power != 1.0:
                dist = np.power(dist, self.tc_fp_power)
            maps.append(dist)
        dist_weight = torch.from_numpy(np.stack(maps, axis=0)).unsqueeze(1)
        dist_weight = dist_weight.to(device=tc_logits.device, dtype=tc_logits.dtype)
        probs = torch.sigmoid(tc_logits)
        return self.tc_fp_weight * (probs * dist_weight).sum() / dist_weight.sum().clamp_min(self.smooth)

    def _single_loss(self, output, target, include_tc_fp):
        regions = self._as_region_target(target).to(device=output.device, dtype=output.dtype)
        regions = self._match_target_size(regions, output.shape[2:])
        wt = regions[:, 0:1]
        tc = regions[:, 1:2]
        et = regions[:, 2:3]
        loss = self._region_loss(output[:, 0:1], wt, alpha=0.3, beta=0.7)
        loss = loss + self._region_loss(output[:, 1:2], tc, alpha=0.3, beta=0.7)
        loss = loss + self.et_weight * self._region_loss(output[:, 2:3], et, alpha=0.6, beta=0.4)
        if include_tc_fp:
            loss = loss + self._tc_fp_loss(output[:, 1:2], tc)
        return loss

    def forward(self, outputs, target):
        if not isinstance(outputs, (list, tuple)):
            return self._single_loss(outputs, target, include_tc_fp=True)

        if isinstance(target, (list, tuple)):
            targets = list(target)
        else:
            targets = [target] * len(outputs)
        n = min(len(outputs), len(targets))
        weights = np.array([1.0 / (2 ** i) for i in range(n)], dtype=np.float32)
        if n > 1:
            weights[-1] = 0.0
        weights = weights / weights.sum()
        total = outputs[0].sum() * 0.0
        for i in range(n):
            total = total + float(weights[i]) * self._single_loss(
                outputs[i],
                targets[i],
                include_tc_fp=(i == 0),
            )
        return total


class ProposedLossMixin:
    def _build_loss(self):
        return NNUNetRegionBoundaryTCFPLoss(
            boundary_weight=_env_float("NNUNET_PROPOSED_BOUNDARY_WEIGHT", 0.5),
            curvature_weight=_env_float("NNUNET_PROPOSED_CURVATURE_WEIGHT", 2.0),
            et_weight=_env_float("NNUNET_PROPOSED_ET_WEIGHT", 1.5),
            bce_weight=_env_float("NNUNET_PROPOSED_BCE_WEIGHT", 0.5),
            tc_fp_weight=_env_float("NNUNET_PROPOSED_TC_FP_WEIGHT", 0.10),
            tc_fp_distance=_env_float("NNUNET_PROPOSED_TC_FP_DISTANCE", 32.0),
            tc_fp_power=_env_float("NNUNET_PROPOSED_TC_FP_POWER", 1.0),
            tc_fp_min_distance=_env_float("NNUNET_PROPOSED_TC_FP_MIN_DISTANCE", 0.0),
        )


class MatchedNNUNetRegionBoundaryTCFPLoss(NNUNetRegionBoundaryTCFPLoss):
    """Proposed loss with the reference trainer's explicit DS aggregation.

    The historical custom loss used ``min(len(outputs), len(targets))`` and
    then zeroed the last active entry.  This opt-in subclass leaves the
    mathematical single-scale loss untouched and only aligns the four MoLG
    outputs with the five planned supervision scales.  The fifth planned
    target is the only permitted unmatched target because its reference
    weight is zero.
    """

    REFERENCE_DS_WEIGHTS = (8.0 / 15.0, 4.0 / 15.0, 2.0 / 15.0, 1.0 / 15.0, 0.0)

    @classmethod
    def _reference_weights(cls, n_outputs, n_targets):
        if n_targets not in (4, 5):
            raise ValueError(
                "Expected four active or five planned DS targets, "
                f"got {n_targets} targets for {n_outputs} outputs"
            )
        if n_outputs != 4:
            raise ValueError(f"Expected four MoLG outputs, got {n_outputs}")
        if n_targets == 5:
            weights = np.asarray(cls.REFERENCE_DS_WEIGHTS, dtype=np.float32)
        else:
            weights = np.asarray(cls.REFERENCE_DS_WEIGHTS[:4], dtype=np.float32)
        if np.any(weights[:n_outputs] <= 0.0):
            raise ValueError(f"A MoLG output was assigned a non-positive DS weight: {weights}")
        return weights

    @staticmethod
    def _validate_alignment(outputs, targets, weights):
        if len(targets) == 5 and weights[-1] != 0.0:
            raise ValueError("The unmatched fifth planned target must have zero weight")
        for index, (output, target) in enumerate(zip(outputs, targets)):
            if output.ndim != 5:
                raise ValueError(f"Output {index} must be 5D, got {tuple(output.shape)}")
            if not isinstance(target, torch.Tensor) or target.ndim not in (4, 5):
                raise ValueError(f"Target {index} has unsupported shape/type: {type(target)}")
            target_spatial = tuple(target.shape[-3:])
            if tuple(output.shape[-3:]) != target_spatial:
                raise ValueError(
                    "Positive-weight DS output/target scale mismatch at "
                    f"index {index}: output={tuple(output.shape)}, "
                    f"target={tuple(target.shape)}"
                )
            if output.shape[0] != target.shape[0]:
                raise ValueError(
                    f"DS batch mismatch at index {index}: "
                    f"output={output.shape[0]}, target={target.shape[0]}"
                )
            if output.shape[1] != 3:
                raise ValueError(
                    f"MoLG region output must have three channels at index {index}, "
                    f"got {output.shape[1]}"
                )

    def forward(self, outputs, target):
        if not isinstance(outputs, (list, tuple)):
            return self._single_loss(outputs, target, include_tc_fp=True)

        outputs = list(outputs)
        if not outputs:
            raise ValueError("Matched proposed loss received no outputs")
        targets = list(target) if isinstance(target, (list, tuple)) else [target] * len(outputs)
        if len(targets) < len(outputs):
            raise ValueError(
                f"Positive-weight output has no target: {len(outputs)} outputs, {len(targets)} targets"
            )
        if len(targets) > len(outputs):
            # Only the reference's fifth, zero-weight target may be unmatched.
            if len(targets) != 5 or len(outputs) != 4:
                raise ValueError(
                    f"Unmatched positive-weight DS target: {len(outputs)} outputs, {len(targets)} targets"
                )
            if not isinstance(targets[-1], torch.Tensor):
                raise ValueError("The unmatched fifth planned target is not a tensor")
        weights = self._reference_weights(len(outputs), len(targets))
        self._validate_alignment(outputs, targets[:len(outputs)], weights)

        total = outputs[0].sum() * 0.0
        for index, output in enumerate(outputs):
            total = total + float(weights[index]) * self._single_loss(
                output,
                targets[index],
                include_tc_fp=(index == 0),
            )
        return total


class ProposedLossDSMatchedMixin:
    def _build_loss(self):
        return MatchedNNUNetRegionBoundaryTCFPLoss(
            boundary_weight=_env_float("NNUNET_PROPOSED_BOUNDARY_WEIGHT", 0.5),
            curvature_weight=_env_float("NNUNET_PROPOSED_CURVATURE_WEIGHT", 2.0),
            et_weight=_env_float("NNUNET_PROPOSED_ET_WEIGHT", 1.5),
            bce_weight=_env_float("NNUNET_PROPOSED_BCE_WEIGHT", 0.5),
            tc_fp_weight=_env_float("NNUNET_PROPOSED_TC_FP_WEIGHT", 0.10),
            tc_fp_distance=_env_float("NNUNET_PROPOSED_TC_FP_DISTANCE", 32.0),
            tc_fp_power=_env_float("NNUNET_PROPOSED_TC_FP_POWER", 1.0),
            tc_fp_min_distance=_env_float("NNUNET_PROPOSED_TC_FP_MIN_DISTANCE", 0.0),
        )


class nnUNetTrainerProposedLoss(ProposedLossMixin, nnUNetTrainer):
    """Official nnU-Net architecture trained with the proposed boundary/TC-FP loss."""

    pass


class nnUNetTrainerSeeded(SeededTrainerMixin, nnUNetTrainer):
    """PlainConvUNet repeat with its initialization seed read from NNUNET_RUN_SEED."""

    pass


class nnUNetTrainerProposedLoss_100ep(nnUNetTrainerProposedLoss):
    """Short screening variant of official nnU-Net with the proposed loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = _num_epochs(100)


class nnUNetTrainerModalSwinUNet(nnUNetTrainer):
    """
    nnU-Net v2 trainer using:
    modality-specific shallow CNN stems + shared CNN U-Net + Swin bottleneck.

    This is scheme A from the design note. It keeps the nnU-Net v2 pipeline
    intact and only replaces the network architecture.
    """

    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=True,
    ):
        return _build_network(
            ModalSwinUNet,
            arch_init_kwargs,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        """Adapt nnU-Net's decoder-specific toggle to the custom network API."""
        network = self.network.module if self.is_ddp else self.network
        # torch.compile wraps the original module in an OptimizedModule.
        network = getattr(network, "_orig_mod", network)
        network.deep_supervision = bool(enabled)


class nnUNetTrainerModalSwinUNet100ep(nnUNetTrainerModalSwinUNet):
    """100-epoch screening variant for fast architecture checks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = _num_epochs(100)


class nnUNetTrainerModalAttnSwinUNet(nnUNetTrainerModalSwinUNet):
    """Scheme A with explicit voxel-wise modality attention."""

    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=True,
    ):
        return _build_network(
            ModalAttnSwinUNet,
            arch_init_kwargs,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )


class nnUNetTrainerModalAttnSwinUNet100ep(nnUNetTrainerModalAttnSwinUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = _num_epochs(100)


class nnUNetTrainerParallelModalSwinUNet(nnUNetTrainerModalSwinUNet):
    """Scheme B: parallel CNN and Swin encoders with scale-wise fusion."""

    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=True,
    ):
        return _build_network(
            ParallelModalSwinUNet,
            arch_init_kwargs,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )


class nnUNetTrainerParallelModalSwinUNet100ep(nnUNetTrainerParallelModalSwinUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = _num_epochs(100)


class nnUNetTrainerParallelModalSwinUNetB24(nnUNetTrainerParallelModalSwinUNet):
    """Paper MoLG architecture (base channels 24) under the full nnU-Net pipeline."""

    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=True,
    ):
        return _build_network(
            ParallelModalSwinUNet,
            arch_init_kwargs,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            base_channels=24,
        )


class nnUNetTrainerParallelModalSwinUNetB24Seeded(
        SeededTrainerMixin,
        nnUNetTrainerParallelModalSwinUNetB24,
):
    """MoLG-B24 repeat with its initialization seed read from NNUNET_RUN_SEED."""

    pass


class nnUNetTrainerParallelModalSwinUNetB24NoMAF(
        SeededTrainerMixin,
        nnUNetTrainerModalSwinUNet,
):
    """Main-pipeline ablation replacing MAF with an early-concat convolution."""

    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=True,
    ):
        return _build_network(
            ParallelSwinUNetNoMAF,
            arch_init_kwargs,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            base_channels=24,
        )


class nnUNetTrainerParallelModalSwinUNetB24NoSwin(
        SeededTrainerMixin,
        nnUNetTrainerModalSwinUNet,
):
    """Main-pipeline ablation removing the complete global Swin branch."""

    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=True,
    ):
        return _build_network(
            ParallelModalUNetNoSwin,
            arch_init_kwargs,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            base_channels=24,
        )


class nnUNetTrainerParallelModalSwinUNetB24ConcatFusion(
        SeededTrainerMixin,
        nnUNetTrainerModalSwinUNet,
):
    """Main-pipeline ablation replacing gated fusion with concat plus 1x1x1 conv."""

    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=True,
    ):
        return _build_network(
            ParallelModalSwinUNetConcatFusion,
            arch_init_kwargs,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            base_channels=24,
        )


class nnUNetTrainerParallelModalSwinUNetB24_100ep(nnUNetTrainerParallelModalSwinUNetB24):
    """Short screening variant of the paper MoLG architecture."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = _num_epochs(100)


class nnUNetTrainerParallelModalSwinUNetB24ProposedLoss(
        ProposedLossMixin,
        nnUNetTrainerParallelModalSwinUNetB24,
):
    """MoLG-B24 architecture trained with the proposed boundary/TC-FP loss."""

    pass


class nnUNetTrainerParallelModalSwinUNetB24ProposedLossDSMatchedSeed42(
        SeededTrainerMixin,
        ProposedLossDSMatchedMixin,
        nnUNetTrainerParallelModalSwinUNetB24,
):
    """Seed42 closeout trainer with the reference-matched proposed DS path."""

    pass


class nnUNetTrainerParallelModalSwinUNetB24ProposedLoss_100ep(
        nnUNetTrainerParallelModalSwinUNetB24ProposedLoss,
):
    """Short screening variant of MoLG-B24 with the proposed loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = _num_epochs(100)


class nnUNetTrainerSearchedModalSwinUNet(nnUNetTrainerModalSwinUNet):
    """
    Scheme A with a CardiacNAS-style searched global bottleneck.

    The search is performed outside nnU-Net. Set NNUNET_SEARCH_ARCH to the
    exported best_arch_GLOBAL.json before starting nnUNetv2_train.
    """

    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=True,
    ):
        return _build_network(
            SearchedModalSwinUNet,
            arch_init_kwargs,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            search_arch=_searched_arch_path(),
        )


class nnUNetTrainerSearchedModalSwinUNet100ep(nnUNetTrainerSearchedModalSwinUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = _num_epochs(100)


class nnUNetTrainerSearchedModalAttnSwinUNet(nnUNetTrainerModalSwinUNet):
    """Recommended model with modality attention and searched global bottleneck."""

    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=True,
    ):
        return _build_network(
            SearchedModalAttnSwinUNet,
            arch_init_kwargs,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            search_arch=_searched_arch_path(),
        )


class nnUNetTrainerSearchedModalAttnSwinUNet100ep(nnUNetTrainerSearchedModalAttnSwinUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = _num_epochs(100)


class nnUNetTrainerSearchedParallelModalSwinUNet(nnUNetTrainerModalSwinUNet):
    """Scheme B with searched Swin blocks at the global branch stages."""

    @staticmethod
    def build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=True,
    ):
        return _build_network(
            SearchedParallelModalSwinUNet,
            arch_init_kwargs,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
            search_arch=_searched_arch_path(),
        )


class nnUNetTrainerSearchedParallelModalSwinUNet100ep(nnUNetTrainerSearchedParallelModalSwinUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = _num_epochs(100)
