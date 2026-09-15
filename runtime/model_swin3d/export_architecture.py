"""
Enhanced Architecture Exporter for SwinUNet3D_NAS
--------------------------------------------------
This script exports the best architecture found by FedNAS, including:

Per-block:
    - ffn_choice (argmax)
    - attn_choice (argmax)
    - residual_alpha in {0.25, 0.50, 0.75}
    - gamma_global = (h, k, alpha)
    - block_index

Per-stage summary:
    - stage_id
    - num_blocks
    - blocks: list of block dicts above

Full model summary:
    - dims
    - depths
    - resolution
    - in_channels
    - num_classes
    - total_blocks
    - stage_summaries

JSON saved at: output_path
"""

import json
import torch
import numpy as np


def export_best_architecture(model, output_path="best_arch.json"):
    """
    Extract NAS architecture as a compact CardiacNAS-style genotype:
      gamma_global = (h, k, alpha)

    Args:
        model: SwinUNet3D_NAS instance (with nas=True, alpha already set)
        output_path: path to save JSON
    """
    assert hasattr(model, "_nas_blocks"), \
        "Model does not contain NAS blocks. Did you use SwinUNet3D_NAS (extended)?"
    assert hasattr(model, "alpha_mgr"), \
        "Model has no alpha manager. NAS must be enabled."

    print(f"[Export] Exporting best architecture → {output_path}")

    # force reassign alpha to blocks
    model.set_alpha()

    block_records = []
    block_index = 0

    # collect all candidate names (for readability)
    # ffn op names are in each block via blk.ffn.ops
    # attn candidates via blk.attn.atts
    # but we do not store the actual op modules in JSON, only index & description.

    # iterate through each block
    for blk in model._nas_blocks:
        # ----------------------------------
        # FFN choice
        # ----------------------------------
        if blk.ffn.alpha is not None:
            alpha_ffn = blk.ffn.alpha.detach().cpu().numpy()
            ffn_idx = int(np.argmax(alpha_ffn))
        else:
            ffn_idx = 0
            alpha_ffn = None

        # Extract kernel/local mixing choice names for readability
        ffn_names = getattr(blk.ffn, "op_names", [op.__class__.__name__ for op in blk.ffn.ops])
        ffn_choice_name = ffn_names[ffn_idx]

        # ----------------------------------
        # Attention choice
        # ----------------------------------
        if blk.attn.alpha is not None:
            alpha_attn = blk.attn.alpha.detach().cpu().numpy()
            attn_idx = int(np.argmax(alpha_attn))
        else:
            attn_idx = 0
            alpha_attn = None

        attn_cands = []
        for a in blk.attn.atts:
            attn_cands.append({
                "window_size": a.window_size,  # tuple
                "num_heads": a.num_heads
            })
        attn_choice = attn_cands[attn_idx]

        # ----------------------------------
        # Residual scaling alpha
        # ----------------------------------
        alpha_candidates = list(getattr(blk, "residual_alpha_candidates", [0.5]))
        if getattr(blk, "residual_alpha", None) is not None:
            raw_alpha = blk.residual_alpha.detach().cpu()
            if raw_alpha.numel() == 1:
                residual_alpha = float((0.2 + 0.6 * torch.sigmoid(blk.residual_alpha.squeeze())).item())
                residual_alpha_idx = 0
                alpha_probs = [1.0]
            else:
                alpha_probs_t = torch.softmax(blk.residual_alpha.detach().cpu(), dim=0)
                residual_alpha_idx = int(torch.argmax(alpha_probs_t).item())
                residual_alpha = float(alpha_candidates[residual_alpha_idx])
                alpha_probs = [float(v) for v in alpha_probs_t.tolist()]
        else:
            raw_alpha = torch.zeros(1)
            residual_alpha_idx = 0
            residual_alpha = float(alpha_candidates[0])
            alpha_probs = [1.0]

        # ----------------------------------
        # Save block record
        # ----------------------------------
        gamma_global = {
            "h": int(attn_choice["num_heads"]),
            "k": ffn_choice_name,
            "alpha": residual_alpha,
        }
        block_records.append({
            "block_index": block_index,
            "ffn_idx": ffn_idx,
            "ffn_choice": ffn_choice_name,
            "attn_idx": attn_idx,
            "attn_choice": attn_choice,
            "residual_alpha_idx": residual_alpha_idx,
            "residual_alpha": residual_alpha,
            "residual_alpha_candidates": alpha_candidates,
            "residual_alpha_probs": alpha_probs,
            "gamma_global": gamma_global,
            "resolution": [blk.D, blk.H, blk.W],
            "shift_size": list(blk.shift_size)
        })

        block_index += 1

    # -------------------------
    # Stage-level summary
    # -------------------------
    stage_summaries = []
    idx = 0
    for stage_id, stage in enumerate(model.stages):
        depth = len(stage.blocks)
        stage_blocks = block_records[idx:idx+depth]
        idx += depth

        stage_summaries.append({
            "stage_id": stage_id,
            "num_blocks": depth,
            "blocks": stage_blocks
        })

    # -------------------------
    # Final JSON structure
    # -------------------------
    final_json = {
        "model": {
            "in_channels": model.in_channels,
            "num_classes": model.num_classes,
            "dims": list(model.dims),
            "depths": list(model.depths),
            "resolution": list(model.resolution),
            "total_blocks": len(model._nas_blocks),
            "genotype": "gamma_global=(h,k,alpha)",
        },
        "stages": stage_summaries,
        "description": "Compact CardiacNAS-style genotype export with attention capacity h, local kernel k, and residual scaling alpha."
    }

    # Save JSON
    with open(output_path, "w") as f:
        json.dump(final_json, f, indent=4)

    print(f"[Export] Done. JSON saved → {output_path}")

    # Print architecture summary
    print_architecture_summary(final_json)

    return final_json


def print_architecture_summary(arch_json):
    """
    Print a human-readable summary of the searched architecture.
    Shows per-block choices: ffn_choice, attn_choice, gate_value, gate_keep
    """
    print("\n" + "="*80)
    print("ARCHITECTURE SEARCH RESULTS")
    print("="*80)

    stages = arch_json["stages"]
    model_info = arch_json["model"]

    print(f"\nModel Configuration:")
    print(f"  Input channels: {model_info['in_channels']}")
    print(f"  Output classes: {model_info['num_classes']}")
    print(f"  Dims: {model_info['dims']}")
    print(f"  Depths: {model_info['depths']}")
    print(f"  Resolution: {model_info['resolution']}")
    print(f"  Total blocks: {model_info['total_blocks']}")

    print(f"\n{'='*80}")
    print("Per-Block Search Results:")
    print(f"{'='*80}")

    for stage in stages:
        stage_id = stage["stage_id"]
        num_blocks = stage["num_blocks"]
        blocks = stage["blocks"]

        print(f"\n--- Stage {stage_id} ({num_blocks} blocks) ---")
        print(f"{'Block':<8} {'Kernel k':<16} {'Attn h':<8} {'Win':<15} {'Alpha':<8} {'gamma_global':<30}")
        print("-" * 80)

        for blk in blocks:
            block_idx = blk["block_index"]
            ffn_choice = blk["ffn_choice"]
            attn = blk["attn_choice"]
            win_str = f"{attn['window_size']}"
            alpha_val = blk["residual_alpha"]
            gamma = blk["gamma_global"]
            gamma_str = f"({gamma['h']},{gamma['k']},{gamma['alpha']:.2f})"

            print(f"{block_idx:<8} {ffn_choice:<16} {attn['num_heads']:<8} {win_str:<15} {alpha_val:<8.2f} {gamma_str:<30}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("Summary Statistics:")
    print(f"{'='*80}")

    all_blocks = []
    for stage in stages:
        all_blocks.extend(stage["blocks"])

    # FFN choices distribution
    ffn_counts = {}
    for blk in all_blocks:
        ffn = blk["ffn_choice"]
        ffn_counts[ffn] = ffn_counts.get(ffn, 0) + 1

    print(f"\nFFN Choices Distribution:")
    for ffn, count in sorted(ffn_counts.items()):
        pct = count / len(all_blocks) * 100
        print(f"  {ffn:<20}: {count:>3} blocks ({pct:>5.1f}%)")

    # Attention choices distribution
    attn_counts = {}
    for blk in all_blocks:
        attn = blk["attn_choice"]
        attn_key = f"win{attn['window_size']}_h{attn['num_heads']}"
        attn_counts[attn_key] = attn_counts.get(attn_key, 0) + 1

    print(f"\nAttention Choices Distribution:")
    for attn, count in sorted(attn_counts.items()):
        pct = count / len(all_blocks) * 100
        print(f"  {attn:<20}: {count:>3} blocks ({pct:>5.1f}%)")

    # Residual scaling statistics
    avg_alpha_val = sum(blk["residual_alpha"] for blk in all_blocks) / len(all_blocks)
    alpha_counts = {}
    for blk in all_blocks:
        alpha_key = f"{blk['residual_alpha']:.2f}"
        alpha_counts[alpha_key] = alpha_counts.get(alpha_key, 0) + 1

    print(f"\nResidual Alpha Statistics:")
    for alpha_key, count in sorted(alpha_counts.items()):
        pct = count / len(all_blocks) * 100
        print(f"  alpha={alpha_key:<5}: {count:>3} blocks ({pct:>5.1f}%)")
    print(f"  Average alpha value: {avg_alpha_val:.4f}")

    print(f"\n{'='*80}\n")


# ------------------------------------------------------------
# Legacy function for backward compatibility
# ------------------------------------------------------------
def export_best_model(model, json_path="best_architecture.json"):
    """
    Legacy function for backward compatibility.
    Extracts best FFN operations from MixedOp3D and exports to JSON.
    Returns a fixed model (if model has _nas_blocks) or just exports JSON.
    """
    import copy
    from .nas_ops import MixedOp3D

    # Check if this is the old-style model (with MixedOp3D in blocks)
    # or new-style model (with _nas_blocks)
    if hasattr(model, "_nas_blocks"):
        # New style: use export_best_architecture
        arch_json = export_best_architecture(model, json_path)
        # For now, return the model itself (caller will handle conversion)
        return model
    else:
        # Old style: extract from MixedOp3D modules
        best_ops = []
        for m in model.modules():
            if isinstance(m, MixedOp3D):
                if m.alpha is None:
                    idx = 0
                else:
                    alpha = m.alpha.detach().cpu()
                    idx = torch.argmax(alpha).item()
                best_ops.append({
                    "index": idx,
                    "op": m.ops[idx].__class__.__name__
                })

        # Export JSON
        arch = {
            "num_mixed_ops": len(best_ops),
            "ops": best_ops
        }
        with open(json_path, "w") as f:
            json.dump(arch, f, indent=4)
        print(f"[OK] Architecture exported to {json_path}")

        # Return model (caller will handle state_dict conversion)
        return model
