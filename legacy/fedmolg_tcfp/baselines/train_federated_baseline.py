import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.models import build_baseline_model
from baselines.train_baseline import (
    evaluate_cases,
    list_cases,
    save_json,
    split_cases,
    summarize_case_metrics,
)
from federated.fl_client import FederatedClient
from federated.fl_server import FederatedServer


def parse_args():
    parser = argparse.ArgumentParser(description="Federated BraTS baseline trainer")
    parser.add_argument("--model", type=str, default="unet3d",
                        choices=["unet3d", "fednas_unet3d", "modal_swin_unet3d",
                                 "modal_attn_swin_unet3d", "parallel_modal_swin_unet3d",
                                 "parallel_modal_raw_swin_unet3d",
                                 "parallel_modal_unet_no_swin3d",
                                 "parallel_swin_unet_no_modal_stem3d",
                                 "search_modal_swin_unet3d", "search_modal_attn_swin_unet3d",
                                 "search_parallel_modal_swin_unet3d",
                                 "searched_modal_swin_unet3d", "searched_modal_attn_swin_unet3d",
                                 "searched_parallel_modal_swin_unet3d",
                                 "segresnet", "fedmunet3d", "mednext_lite",
                                 "benchmark_fets_dynunet", "fets_dynunet",
                                 "monai_segresnet", "monai_swinunetr"])
    parser.add_argument("--exp_name", type=str, default="fedavg_unet3d")
    parser.add_argument("--fl_algorithm", type=str, default="fedavg",
                        choices=["fedavg", "fedprox", "fedbn"],
                        help="Federated optimization baseline to run")
    parser.add_argument("--fedprox_mu", type=float, default=1e-3,
                        help="FedProx proximal coefficient")
    parser.add_argument("--fedbn_norm_scope", type=str, default="all",
                        choices=["batch", "all"],
                        help="FedBN local norm scope: strict BatchNorm only, or all norm layers")
    parser.add_argument("--data_root", type=str, default="dataset/train")
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--nas_op", type=str, default="mixed",
                        choices=["mixed", "conv3", "conv5", "depthwise3",
                                 "residual_depthwise", "residual", "skip"],
                        help="Block operation for fednas_unet3d")
    parser.add_argument("--patch_size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_alpha", type=float, default=0.0,
                        help="Architecture learning rate for federated searchable models")
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss_type", type=str, default="hybrid",
                        choices=["ce", "region", "boundary", "hybrid"],
                        help="region disables boundary/curvature weights; boundary/hybrid enable them")
    parser.add_argument("--boundary_weight", type=float, default=0.5)
    parser.add_argument("--curvature_weight", type=float, default=2.0)
    parser.add_argument("--et_loss_weight", type=float, default=1.5)
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--tc_fp_weight", type=float, default=0.0,
                        help="Weight for distance-aware TC false-positive suppression loss")
    parser.add_argument("--tc_fp_distance", type=float, default=32.0,
                        help="Distance cap in voxels for TC false-positive suppression")
    parser.add_argument("--tc_fp_power", type=float, default=1.0,
                        help="Power applied to normalized TC distance weights")
    parser.add_argument("--tc_fp_min_distance", type=float, default=0.0,
                        help="Do not penalize TC predictions within this distance from GT TC")
    parser.add_argument("--tc_fp_ramp_start_round", type=int, default=0,
                        help="Round where TC false-positive loss starts ramping from zero")
    parser.add_argument("--tc_fp_ramp_end_round", type=int, default=0,
                        help="Round where TC false-positive loss reaches --tc_fp_weight")
    parser.add_argument("--warmup_rounds", type=int, default=50)
    parser.add_argument("--search_warmup_rounds", type=int, default=20,
                        help="Rounds before architecture parameters are updated")
    parser.add_argument("--search_temp_start", type=float, default=5.0)
    parser.add_argument("--search_temp_end", type=float, default=0.2)
    parser.add_argument("--save_top_k", type=int, default=5)
    parser.add_argument("--checkpoint_score", type=str, default="dice",
                        choices=["dice", "dice_hd95", "dice_hd95_params"],
                        help="Top-K averaging score: Dice, Dice-HD95, or Dice-HD95-Params")
    parser.add_argument("--hd95_penalty", type=float, default=0.01)
    parser.add_argument("--params_penalty", type=float, default=0.001,
                        help="Penalty per million parameters for checkpoint_score=dice_hd95_params")
    parser.add_argument("--checkpoint_score_candidates", type=int, default=10,
                        help="Number of Dice-preselected candidates to score with HD95")
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable AMP in clients and final evaluation")
    parser.add_argument("--no_amp", action="store_true", help="Force-disable AMP even on CUDA")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--search_arch", type=str, default="",
                        help="Architecture JSON for searched_* retraining models")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model_fn(args):
    def model_fn():
        return build_baseline_model(
            args.model,
            in_channels=4,
            out_channels=3,
            base_channels=args.base_channels,
            patch_size=tuple(args.patch_size),
            nas_op=args.nas_op,
            search_arch=args.search_arch,
        )

    return model_fn


def make_loss_config(args):
    return {
        "loss_type": args.loss_type,
        "weight_boundary": args.boundary_weight,
        "curvature_weight": args.curvature_weight,
        "et_weight": args.et_loss_weight,
        "bce_weight": args.bce_weight,
        "tc_fp_weight": args.tc_fp_weight,
        "tc_fp_distance": args.tc_fp_distance,
        "tc_fp_power": args.tc_fp_power,
        "tc_fp_min_distance": args.tc_fp_min_distance,
    }


def prepare_datasets(args):
    from datasets.brats_multimodal_dataset import BraTSMultiModalDataset

    cases = list_cases(args.data_root)
    train_cases, test_cases = split_cases(cases, args.test_ratio, args.seed)
    splits = np.array_split(train_cases, args.num_clients)
    train_datasets = [
        BraTSMultiModalDataset(
            list(split),
            target_size=tuple(args.patch_size),
            augment=True,
        )
        for split in splits
    ]
    test_ds = BraTSMultiModalDataset(test_cases, augment=False, test_mode=True)
    return train_datasets, test_ds, test_cases


def print_model_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[Model Stats | Federated Baseline]")
    print(f"  - Total Params:     {total / 1e6:.2f} M")
    print(f"  - Trainable Params: {trainable / 1e6:.2f} M")
    return total


def current_lr(args, round_idx):
    progress = round_idx / max(1, args.rounds)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def current_search_temp(args, round_idx):
    progress = round_idx / max(1, args.rounds - 1)
    return args.search_temp_end + 0.5 * (args.search_temp_start - args.search_temp_end) * (
        1 + math.cos(math.pi * progress)
    )


def current_tc_fp_weight(args, round_idx):
    target_weight = float(args.tc_fp_weight)
    if target_weight <= 0.0:
        return 0.0

    start = int(args.tc_fp_ramp_start_round)
    end = int(args.tc_fp_ramp_end_round)
    current_round = round_idx + 1
    if end <= start:
        return target_weight
    if current_round <= start:
        return 0.0
    if current_round >= end:
        return target_weight

    progress = (current_round - start) / max(1, end - start)
    return target_weight * progress


def norm_module_types(scope):
    if scope == "batch":
        return (nn.modules.batchnorm._BatchNorm,)

    types = [nn.modules.batchnorm._BatchNorm, nn.GroupNorm, nn.LayerNorm]
    instance_norm = getattr(nn.modules.instancenorm, "_InstanceNorm", None)
    if instance_norm is not None:
        types.append(instance_norm)
    return tuple(types)


def get_norm_state_keys(model, scope="all"):
    norm_types = norm_module_types(scope)
    keys = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, norm_types):
            continue
        prefix = f"{module_name}." if module_name else ""
        for local_key in module.state_dict().keys():
            keys.add(prefix + local_key)
        for local_key, _ in module.named_parameters(recurse=False):
            keys.add(prefix + local_key)
        for local_key, _ in module.named_buffers(recurse=False):
            keys.add(prefix + local_key)
    return keys


def clone_state_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return value


def clone_state_dict(state):
    return {key: clone_state_value(value) for key, value in state.items()}


def filter_state_dict(state, excluded_keys):
    if not excluded_keys:
        return state
    return {key: value for key, value in state.items() if key not in excluded_keys}


def capture_client_norm_states(clients, norm_keys):
    norm_states = []
    for client in clients:
        state = client.model.state_dict()
        norm_states.append({
            key: clone_state_value(state[key])
            for key in norm_keys
            if key in state
        })
    return norm_states


def save_fedbn_checkpoint(server, clients, norm_keys, path, round_idx, val_dice):
    raw_state = server.global_model.state_dict()
    model_state = filter_state_dict(raw_state, {k for k in raw_state if k.endswith(".alpha")})
    torch.save(
        {
            "model": clone_state_dict(model_state),
            "alpha": getattr(server, "global_alphas", []),
            "client_norm_states": capture_client_norm_states(clients, norm_keys),
            "norm_keys": sorted(norm_keys),
            "round": round_idx,
            "val_dice": val_dice,
            "fl_algorithm": "fedbn",
        },
        path,
    )


def checkpoint_pool_limit(args):
    if args.checkpoint_score in {"dice_hd95", "dice_hd95_params"}:
        return max(args.save_top_k, args.checkpoint_score_candidates)
    return args.save_top_k


def save_if_top_k(server, saved, round_idx, val_dice, ckpt_dir, args, clients=None, norm_keys=None):
    if round_idx <= args.warmup_rounds:
        return saved

    pool_limit = checkpoint_pool_limit(args)
    should_save = len(saved) < pool_limit or val_dice > saved[-1][0]
    if not should_save:
        return saved

    ckpt_path = ckpt_dir / f"round_{round_idx:03d}_dice_{val_dice:.4f}.pth"
    if args.fl_algorithm == "fedbn":
        save_fedbn_checkpoint(server, clients, norm_keys or set(), ckpt_path, round_idx, val_dice)
    else:
        server.save(str(ckpt_path))
    saved.append((val_dice, str(ckpt_path)))
    saved.sort(key=lambda x: x[0], reverse=True)

    while len(saved) > pool_limit:
        _, old_path = saved.pop(-1)
        if os.path.exists(old_path):
            os.remove(old_path)

    print(f"  [TopK] Saved {ckpt_path}")
    return saved


def evaluate_checkpoint_metrics(clients, ckpt_path, device, loss_config):
    state = torch.load(ckpt_path, map_location=device)
    if "model" in state:
        state = state["model"]

    all_metrics = []
    for client in clients:
        metrics_res = client.evaluate(state, loss_config=loss_config)
        if metrics_res:
            all_metrics.append(metrics_res)

    if not all_metrics:
        return {}

    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
    return avg_metrics


def select_top_checkpoints(saved, clients, device, args):
    if args.checkpoint_score == "dice":
        return [
            {
                "score": dice,
                "dice": dice,
                "hd95": None,
                "path": path,
            }
            for dice, path in sorted(saved, key=lambda x: x[0], reverse=True)[:args.save_top_k]
        ]

    if args.fl_algorithm == "fedbn":
        print("[Checkpoint Selection] HD95-based scoring is not enabled for FedBN local norm states; using Dice ranking.")
        return select_top_checkpoints(saved, clients, device, argparse.Namespace(**{
            **vars(args),
            "checkpoint_score": "dice",
        }))

    preselect_count = max(args.save_top_k, args.checkpoint_score_candidates)
    preselected = sorted(saved, key=lambda x: x[0], reverse=True)[:preselect_count]
    scored = []
    loss_config = make_loss_config(args)
    params_m = 0.0
    if args.checkpoint_score == "dice_hd95_params" and clients:
        params_m = sum(p.numel() for p in clients[0].model.parameters()) / 1e6

    print(
        f"[Checkpoint Selection] Scoring top-{len(preselected)} candidates with "
        f"Dice - {args.hd95_penalty:g} * HD95"
        + (f" - {args.params_penalty:g} * Params_M" if args.checkpoint_score == "dice_hd95_params" else "")
    )
    for dice, path in preselected:
        metrics_res = evaluate_checkpoint_metrics(clients, path, device, loss_config)
        dice_mean = metrics_res.get("Dice_Mean", dice)
        hd95_mean = metrics_res.get("HD95_Mean", 0.0)
        score = dice_mean - args.hd95_penalty * hd95_mean
        if args.checkpoint_score == "dice_hd95_params":
            score = score - args.params_penalty * params_m
        scored.append({
            "score": score,
            "dice": dice_mean,
            "hd95": hd95_mean,
            "params_m": params_m if args.checkpoint_score == "dice_hd95_params" else None,
            "path": path,
        })
        if args.checkpoint_score == "dice_hd95_params":
            print(
                f"  - Score: {score:.4f} | Dice: {dice_mean:.4f} | "
                f"HD95: {hd95_mean:.4f} | Params_M: {params_m:.2f} | Path: {path}"
            )
        else:
            print(f"  - Score: {score:.4f} | Dice: {dice_mean:.4f} | HD95: {hd95_mean:.4f} | Path: {path}")

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:args.save_top_k]


def average_floating_state_dicts(states):
    avg_state = clone_state_dict(states[0])
    for state in states[1:]:
        for key, value in state.items():
            if key in avg_state and isinstance(avg_state[key], torch.Tensor) and avg_state[key].is_floating_point():
                avg_state[key] = avg_state[key] + value.detach().cpu()
    for key, value in avg_state.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            avg_state[key] = value / len(states)
    return avg_state


def average_client_norm_states(checkpoints):
    first = checkpoints[0].get("client_norm_states", [])
    if not first:
        return []

    averaged = [clone_state_dict(client_state) for client_state in first]
    for ckpt in checkpoints[1:]:
        for cid, client_state in enumerate(ckpt.get("client_norm_states", [])):
            if cid >= len(averaged):
                continue
            for key, value in client_state.items():
                if (key in averaged[cid]
                        and isinstance(averaged[cid][key], torch.Tensor)
                        and averaged[cid][key].is_floating_point()):
                    averaged[cid][key] = averaged[cid][key] + value.detach().cpu()

    for client_state in averaged:
        for key, value in client_state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                client_state[key] = value / len(checkpoints)
    return averaged


def average_alpha_lists(checkpoints):
    alpha_lists = [ckpt.get("alpha", []) for ckpt in checkpoints if ckpt.get("alpha", [])]
    if not alpha_lists:
        return []
    num_alpha = min(len(alpha) for alpha in alpha_lists)
    averaged = []
    for idx in range(num_alpha):
        vals = [alpha[idx].detach().cpu() for alpha in alpha_lists]
        avg = vals[0].clone()
        for val in vals[1:]:
            avg = avg + val
        averaged.append(avg / len(vals))
    return averaged


def average_top_k(server, selected, avg_path, device, args):
    if not selected:
        raise RuntimeError("No checkpoints were saved; lower warmup_rounds or increase rounds.")

    print(f"[Final] Averaging top-{len(selected)} checkpoints:")
    paths = []
    for item in selected:
        if isinstance(item, dict):
            paths.append(item["path"])
            if item["hd95"] is None:
                print(f"  - Dice: {item['dice']:.4f} | Path: {item['path']}")
            else:
                print(
                    f"  - Score: {item['score']:.4f} | Dice: {item['dice']:.4f} | "
                    f"HD95: {item['hd95']:.4f}"
                    + (f" | Params_M: {item['params_m']:.2f}" if item.get("params_m") is not None else "")
                    + f" | Path: {item['path']}"
                )
        else:
            dice, path = item
            paths.append(path)
            print(f"  - Dice: {dice:.4f} | Path: {path}")

    checkpoints = [torch.load(path, map_location=device) for path in paths]
    avg_state = average_floating_state_dicts([ckpt["model"] for ckpt in checkpoints])
    avg_alpha = average_alpha_lists(checkpoints) or getattr(server, "global_alphas", [])
    if avg_alpha:
        server.global_alphas = [alpha.to(device) for alpha in avg_alpha]
        server.apply_alpha_to_model()

    server.global_model.load_state_dict(avg_state, strict=False)
    payload = {
        "model": avg_state,
        "alpha": avg_alpha,
        "fl_algorithm": args.fl_algorithm,
    }
    if args.fl_algorithm == "fedbn":
        payload["client_norm_states"] = average_client_norm_states(checkpoints)
        payload["norm_keys"] = checkpoints[0].get("norm_keys", [])
    torch.save(payload, str(avg_path))
    print(f"[*] Averaged model saved to {avg_path}")
    return avg_path


def load_checkpoint_to_model(model, checkpoint, device, client_norm_state=None):
    state = torch.load(checkpoint, map_location=device)
    alpha_state = state.get("alpha", []) if isinstance(state, dict) else []
    if "model" in state:
        state = dict(state["model"])
    if client_norm_state:
        state.update(client_norm_state)
    model.load_state_dict(state, strict=False)
    if alpha_state and hasattr(model, "arch_parameters"):
        with torch.no_grad():
            for param, alpha in zip(model.arch_parameters(), alpha_state):
                param.copy_(alpha.to(device=param.device, dtype=param.dtype))


def evaluate_fedbn_cases(model_fn, model_path, test_cases, args):
    from datasets.brats_multimodal_dataset import BraTSMultiModalDataset

    checkpoint = torch.load(model_path, map_location=args.device)
    client_norm_states = checkpoint.get("client_norm_states", [])
    if not client_norm_states:
        raise RuntimeError("FedBN checkpoint does not contain client_norm_states.")

    case_metrics = []
    test_splits = np.array_split(np.array(test_cases, dtype=object), args.num_clients)
    for cid, split in enumerate(test_splits):
        split_cases = list(split)
        if not split_cases:
            continue
        norm_state = client_norm_states[min(cid, len(client_norm_states) - 1)]
        model = model_fn().to(args.device)
        load_checkpoint_to_model(model, model_path, args.device, client_norm_state=norm_state)
        test_ds = BraTSMultiModalDataset(split_cases, augment=False, test_mode=True)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
        print(f"[FedBN Eval] Client-norm {cid}: {len(split_cases)} held-out cases")
        case_metrics.extend(
            evaluate_cases(
                model,
                test_loader,
                split_cases,
                args,
                compute_hd95=True,
            )
        )
    return case_metrics


def final_evaluation(model_fn, model_path, test_ds, test_cases, params_m, args, exp_dir):
    if args.fl_algorithm == "fedbn":
        case_metrics = evaluate_fedbn_cases(model_fn, model_path, test_cases, args)
    else:
        model = model_fn().to(args.device)
        load_checkpoint_to_model(model, model_path, args.device)
        if hasattr(model, "export_search_architecture"):
            save_path = exp_dir / "best_arch_GLOBAL.json"
            model.export_search_architecture(save_path)
            print(f"[Architecture] Exported federated searched architecture -> {save_path}")
        if hasattr(model, "export_architecture"):
            save_json(exp_dir / "fednas_architecture.json", model.export_architecture())
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
        case_metrics = evaluate_cases(
            model,
            test_loader,
            test_cases,
            args,
            compute_hd95=True,
        )
    summary = summarize_case_metrics(case_metrics)
    summary["Params_M"] = params_m
    summary["Training"] = {
        "environment": "federated",
        "fl_algorithm": args.fl_algorithm,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "model": args.model,
        "base_channels": args.base_channels,
        "nas_op": args.nas_op if args.model == "fednas_unet3d" else None,
        "search_arch": args.search_arch or None,
        "lr_alpha": args.lr_alpha if args.lr_alpha > 0 else None,
        "search_warmup_rounds": args.search_warmup_rounds if args.lr_alpha > 0 else None,
        "loss_type": args.loss_type,
        "boundary_weight": args.boundary_weight,
        "curvature_weight": args.curvature_weight,
        "tc_fp_weight": args.tc_fp_weight if args.tc_fp_weight > 0 else None,
        "tc_fp_distance": args.tc_fp_distance if args.tc_fp_weight > 0 else None,
        "tc_fp_power": args.tc_fp_power if args.tc_fp_weight > 0 else None,
        "tc_fp_min_distance": args.tc_fp_min_distance if args.tc_fp_weight > 0 else None,
        "tc_fp_ramp_start_round": args.tc_fp_ramp_start_round if args.tc_fp_weight > 0 else None,
        "tc_fp_ramp_end_round": args.tc_fp_ramp_end_round if args.tc_fp_weight > 0 else None,
        "checkpoint_score": args.checkpoint_score,
        "hd95_penalty": args.hd95_penalty if args.checkpoint_score != "dice" else None,
        "params_penalty": args.params_penalty if args.checkpoint_score == "dice_hd95_params" else None,
        "fedprox_mu": args.fedprox_mu if args.fl_algorithm == "fedprox" else None,
        "fedbn_norm_scope": args.fedbn_norm_scope if args.fl_algorithm == "fedbn" else None,
    }
    save_json(exp_dir / "case_metrics.json", case_metrics)
    save_json(exp_dir / "final_metrics.json", summary)

    print("========== FINAL FEDERATED BASELINE SUMMARY ==========")
    for key, value in summary.items():
        if isinstance(value, dict) and "mean" in value:
            print(f"{key}: {value['mean']:.4f} ± {value['std']:.4f}")
        else:
            print(f"{key}: {value}")


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    args.amp = bool(args.amp and not args.no_amp and args.device == "cuda")

    exp_dir = Path("results") / args.exp_name
    ckpt_dir = Path("checkpoints") / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"========== FEDERATED BASELINE: {args.exp_name} ({args.model}, {args.fl_algorithm}) ==========")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    train_datasets, test_ds, test_cases = prepare_datasets(args)
    model_fn = build_model_fn(args)
    server = FederatedServer(model_fn, args.num_clients, device=args.device)
    total_params = print_model_parameters(server.global_model)
    params_m = total_params / 1e6

    amp_enabled = args.amp
    clients = [
        FederatedClient(
            cid=i,
            model_fn=model_fn,
            dataset=train_datasets[i],
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.local_epochs,
            lr=args.lr,
            lr_alpha=args.lr_alpha,
            weight_decay=args.weight_decay,
            val_split_ratio=args.val_split_ratio,
            num_workers=args.num_workers,
            use_amp=amp_enabled,
        )
        for i in range(args.num_clients)
    ]

    norm_keys = set()
    if args.fl_algorithm == "fedbn":
        norm_keys = get_norm_state_keys(server.global_model, args.fedbn_norm_scope)
        initial_state = server.global_model.state_dict()
        for client in clients:
            client.model.load_state_dict(initial_state, strict=False)
        print(f"[FedBN] Keeping {len(norm_keys)} normalization state entries local "
              f"(scope={args.fedbn_norm_scope}).")

    avg_name = f"{args.fl_algorithm}_final_avg_model.pth"
    if args.checkpoint_score != "dice":
        avg_name = f"{args.fl_algorithm}_final_avg_model_{args.checkpoint_score}.pth"
    avg_path = ckpt_dir / avg_name
    if args.checkpoint:
        avg_path = Path(args.checkpoint)

    if not args.eval_only and not args.checkpoint:
        saved = []
        loss_config = make_loss_config(args)
        if args.fl_algorithm == "fedprox":
            loss_config["fedprox_mu"] = args.fedprox_mu
        for rnd in range(1, args.rounds + 1):
            lr = current_lr(args, rnd - 1)
            temp = current_search_temp(args, rnd - 1)
            round_loss_config = dict(loss_config)
            round_tc_fp_weight = current_tc_fp_weight(args, rnd - 1)
            round_loss_config["tc_fp_weight"] = round_tc_fp_weight
            is_search_warmup = rnd <= args.search_warmup_rounds
            print(
                f"--- {args.fl_algorithm.upper()} Round {rnd}/{args.rounds} "
                f"[LR: {lr:.2e} | Temp: {temp:.3f} | ArchWarmup: {is_search_warmup} "
                f"| TC-FP: {round_tc_fp_weight:.4f}] ---"
            )
            global_weights = server.get_global_weights()
            if args.fl_algorithm == "fedbn":
                global_weights = filter_state_dict(global_weights, norm_keys)
            results = {}
            for cid, client in enumerate(clients):
                results[cid] = client.train(
                    global_weights,
                    getattr(server, "global_alphas", []),
                    current_lr=lr,
                    loss_config=round_loss_config,
                    current_temp=temp,
                    is_warmup=is_search_warmup,
                )

            if args.fl_algorithm == "fedbn":
                weights_list = [filter_state_dict(results[cid]["weights"], norm_keys) for cid in results]
            else:
                weights_list = [results[cid]["weights"] for cid in results]
            sizes = [results[cid]["size"] for cid in results]
            server.set_global_weights(server.aggregate_params(weights_list, sizes))
            server.aggregate_alpha([results[cid].get("alpha", []) for cid in results], sizes)
            server.apply_alpha_to_model()

            train_loss = float(np.mean([results[cid].get("loss", 0.0) for cid in results]))
            val_dice = float(np.mean([results[cid].get("val_dice", 0.0) for cid in results]))
            print(f"  [Log] Round {rnd} Result -> Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")
            saved = save_if_top_k(
                server,
                saved,
                rnd,
                val_dice,
                ckpt_dir,
                args,
                clients=clients,
                norm_keys=norm_keys,
            )

        selected = select_top_checkpoints(saved, clients, args.device, args)
        avg_path = average_top_k(server, selected, avg_path, args.device, args)

    final_evaluation(model_fn, str(avg_path), test_ds, test_cases, params_m, args, exp_dir)


if __name__ == "__main__":
    main()
