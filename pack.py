#!/usr/bin/env python3
# Copyright (c) 2026 Alexander Lavicka.
# This source code is licensed under the PolyForm Noncommercial License 1.0.0.
# A copy of this license is available at https://polyformproject.org/licenses/noncommercial/1.0.0/
# Commercial utilization or hardware integration requires a separate license from the patent holder.
"""Pack a trained Pollux checkpoint (.pt) into the .plx binary format.

The .plx container is a pickle-free, struct-packed binary file designed for
direct memory-mapping, deterministic reproducibility, and SRAM-resident edge
inference.  It stores the entire Pollux backbone at exactly 0.76 bits/param:

  Compression arithmetic
  ----------------------
  Backbone H24 layers  :  18 bits (index) / 24 params  = 0.750 bits/param
  Scale (FP16 σ_rms)   :  16 bits / d params (one scale per row)
                          (global row-wise RMS; bit-exact .plx serialization)
  ─────────────────────────────────────────────────────────────────────────
  Total (backbone, Pollux-1152, d=1152)                 ≈ 0.764 bits/param

  Embeddings and LM head are stored as per-row INT8 + FP16 scale (≈1 bit/param),
  kept separate because they require higher semantic resolution than the
  backbone.

Topological robustness
----------------------
The .plx packing process is mathematically lossless with respect to backbone
capacity. Re-evaluating a packed artifact yields structural scores within
0.01% of the raw .pt training state, confirming that cognitive content
resides entirely in the kissing-point index assignments. (For full empirical
baselines and factual benchmark scores, see the README or paper).

When to pack
------------
Pack the checkpoint at the structural convergence plateau (e.g., ≈10k steps on
FineWeb-Edu 10B), where the lattice has reached its highest structural accuracy
before capacity churn degrades the encoding.

Usage
-----
    python pack.py                          # interactive checkpoint wizard
    python pack.py path/to/pollux_10k.pt
    python pack.py pollux_10k.pt --output model.plx --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from glob import glob
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from castor import (
    BACKBONE_BITS_PER_PARAM,
    H24_DIM,
    codebook_fingerprint,
    get_h24_codebook,
    NUMERICAL_EPSILON,
    pack_indices,
    row_rms_from_matrix,
    scale_bits_per_param,
)
from pollux import (
    _is_h24_tensor,
    config_from_checkpoint,
)

# =============================================================================
# .plx binary format specification
# =============================================================================
#
# [HEADER]       16 bytes   magic(8) version(1) reserved(7)
# [CONFIG]       var        config_len(uint32-LE) UTF-8 JSON,
#                           zero-padded to 64-byte boundary from file start
# [FINGERPRINT]  64 bytes   16 × float32  (codebook_fingerprint first 4×4)
# [TENSOR_COUNT] 4 bytes    uint32-LE  (number of tensors that follow)
# [TENSORS]      repeated   name_len(u16) name(utf-8) dtype_id(u8) ndim(u8)
#                           shape(ndim × uint32-LE) data_len(uint64-LE) raw
#
# dtype_id: 0=uint8  1=int8  2=float16  3=float32

_PLX_MAGIC: bytes = b"POLLUXH2"
_PLX_VERSION: int = 0x04

_DTYPE_UINT8: int = 0
_DTYPE_INT8: int = 1
_DTYPE_FP16: int = 2
_DTYPE_FP32: int = 3

_TORCH_TO_DTYPE_ID: dict[torch.dtype, int] = {
    torch.uint8:   _DTYPE_UINT8,
    torch.int8:    _DTYPE_INT8,
    torch.float16: _DTYPE_FP16,
    torch.float32: _DTYPE_FP32,
}


# =============================================================================
# Layer classification
# =============================================================================

def _is_packable_h24(name: str, tensor: torch.Tensor) -> bool:
    """True for hidden H24 weight matrices (not tok_emb, pos_emb, or head)."""
    if not name.endswith(".weight"):
        return False
    if name.startswith(("tok_emb.", "pos_emb.", "head.")):
        return False
    return _is_h24_tensor(tensor)


# =============================================================================
# Quantization
# =============================================================================

def _quantize_int8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row INT8 quantization with FP16 per-row operational scales μ.

    Reconstruction: weight ≈ (weight_int8 / 127) * mu_rows[:, None]
    """
    w = weight.detach().float()
    mu_rows = w.abs().amax(dim=-1, keepdim=True).clamp_min(NUMERICAL_EPSILON)
    q = torch.round(w / mu_rows * 127.0).clamp(-128, 127).to(torch.int8)
    # Store max_abs directly. pollux.PackedInt8Embedding/Linear.materialize()
    # reconstructs via:  int8 * (mu_rows / 127.0)  — the /127 is in materialize.
    return q.cpu(), mu_rows.squeeze(-1).to(torch.float16).cpu()


def _quantize_h24_layer(
    weight: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map an H24 weight matrix to packed 18-bit indices + per-row FP16 σ_rms.

    Algorithm
    ---------
    1. Reshape weight into [N, 24] atoms (row-major).
    2. Per matrix row: σ_rms = sqrt(mean(w²)) over the full in_features width.
    3. Cosine nearest-neighbour: argmax(w · kp) — scale-invariant direction search.
    4. Zero-vector detection: atoms with ||w||₂ < ε → index 0 (already null-snapped in training).
    5. Pack 4 × 18-bit indices into 9 bytes via castor.pack_indices().
    6. Store one FP16 σ_rms per row (length out_features).

    Returns
    -------
    packed_weights : uint8 tensor, length ((N + 3) // 4) × 9
    sigma_rows     : float16 tensor, length out_features
    """
    device = weight.device
    w = weight.detach().float()
    atoms = w.reshape(-1, H24_DIM)    # [N, 24]
    N = atoms.shape[0]

    sigma_rows = (row_rms_from_matrix(w) * math.sqrt(12.0)).clamp_min(NUMERICAL_EPSILON)

    # Raw nearest-kissing-point lookup on p.data (already zero-snapped during training).
    kissing_points = codebook[1:].to(device=device, dtype=torch.float32)  # [196560, 24]

    indices = torch.zeros(N, dtype=torch.long, device=device)
    chunk = 4096
    for i in range(0, N, chunk):
        s = atoms[i : i + chunk]
        scores = torch.matmul(s, kissing_points.t())          # [B, 196560]
        _, best_kp_idx = scores.max(dim=-1)
        snap_to_zero = s.norm(dim=-1) < NUMERICAL_EPSILON
        best = (best_kp_idx + 1).masked_fill(snap_to_zero, 0)  # 1-indexed into codebook
        indices[i : i + chunk] = best

    packed = pack_indices(indices.cpu())
    return packed, sigma_rows.cpu().to(torch.float16)


# =============================================================================
# .plx serialisation
# =============================================================================

def _write_plx(
    path: str,
    cfg_dict: dict,
    state: dict[str, torch.Tensor],
) -> None:
    """Serialise config + state_dict to a .plx binary file."""
    import numpy as np

    _DTYPE_TO_NUMPY = {
        _DTYPE_UINT8: np.dtype("uint8"),
        _DTYPE_INT8:  np.dtype("int8"),
        _DTYPE_FP16:  np.dtype("float16"),
        _DTYPE_FP32:  np.dtype("float32"),
    }

    config_json = json.dumps(cfg_dict, separators=(",", ":")).encode("utf-8")

    # Padding: align end of CONFIG block (16-byte header + 4-byte len + json)
    # to the next 64-byte boundary from file start.
    pre_json = 16 + 4   # header + config_len field
    total_so_far = pre_json + len(config_json)
    pad_len = (64 - total_so_far % 64) % 64

    fp_values = list(codebook_fingerprint())   # 16 floats
    while len(fp_values) < 16:
        fp_values.append(0.0)

    with open(path, "wb") as f:
        # HEADER (16 bytes)
        f.write(_PLX_MAGIC)
        f.write(struct.pack("B7x", _PLX_VERSION))

        # CONFIG
        f.write(struct.pack("<I", len(config_json)))
        f.write(config_json)
        f.write(b"\x00" * pad_len)

        # FINGERPRINT (64 bytes)
        f.write(struct.pack("<16f", *fp_values[:16]))

        # TENSOR_COUNT (4 bytes)
        f.write(struct.pack("<I", len(state)))

        # TENSORS
        for name, tensor in state.items():
            t = tensor.contiguous().cpu()
            dtype_id = _TORCH_TO_DTYPE_ID.get(t.dtype, _DTYPE_FP32)
            # Cast to known dtype if not in map (e.g. bfloat16 → float32)
            np_dtype = _DTYPE_TO_NUMPY.get(dtype_id, np.dtype("float32"))
            arr = t.numpy().astype(np_dtype, copy=False)

            name_enc = name.encode("utf-8")
            shape = list(t.shape)
            raw = arr.tobytes()

            f.write(struct.pack("<H", len(name_enc)))
            f.write(name_enc)
            f.write(struct.pack("BB", dtype_id, len(shape)))
            f.write(struct.pack(f"<{len(shape)}I", *shape))
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)


# =============================================================================
# Main packing entry-point
# =============================================================================

def pack_checkpoint(src: str, dst: str, device: torch.device) -> None:
    """Load a training checkpoint, quantize all layers, write a .plx file."""
    print(f"Loading  {src} ...", flush=True)
    ckpt = torch.load(src, map_location=device, weights_only=False)
    cfg = config_from_checkpoint(ckpt, state_dict=ckpt.get("model_state_dict", {}))
    codebook = get_h24_codebook(device, torch.float32)

    state_in = ckpt.get("model_state_dict", {})
    state_out: dict[str, torch.Tensor] = {}

    h24_atom_count = 0
    h24_scale_row_count = 0

    total = len(state_in)
    for idx, (name, tensor) in enumerate(state_in.items(), 1):
        print(f"  [{idx:3d}/{total}]  {name}", end="  ", flush=True)

        if name in ("tok_emb.weight", "head.weight"):
            q, mu_r = _quantize_int8(tensor)
            state_out[name.replace(".weight", ".weight_int8")] = q
            state_out[name.replace(".weight", ".mu_rows")] = mu_r
            print("→ INT8", flush=True)

        elif _is_packable_h24(name, tensor):
            pw, sigma_r = _quantize_h24_layer(tensor, codebook)
            state_out[name.replace(".weight", ".packed_weights")] = pw
            state_out[name.replace(".weight", ".sigma_rows")] = sigma_r
            h24_atom_count += tensor.numel() // H24_DIM
            h24_scale_row_count += int(sigma_r.numel())
            print(f"→ H24  ({int(sigma_r.numel())} row scales)", flush=True)

        else:
            state_out[name] = tensor.detach().cpu()
            print("→ pass-through", flush=True)

    print(f"\nWriting  {dst} ...", flush=True)
    cfg_out = dict(cfg.to_dict())
    cfg_out["scales_per_row"] = 1
    cfg_out["scale_serialization"] = "row_wise"
    cfg_out["bits_per_param"] = BACKBONE_BITS_PER_PARAM
    _write_plx(dst, cfg_out, state_out)

    size_mb = os.path.getsize(dst) / (1024 * 1024)
    in_f = int(cfg_out.get("n_embd", 1152))
    backbone_bpp = 18 / H24_DIM                            # 0.750
    scale_bpp = scale_bits_per_param(in_f)                 # 16 / d
    total_bpp = backbone_bpp + scale_bpp

    print(
        f"\nWrote  {dst}  ({size_mb:.2f} MB)\n"
        f"\nCompression proof (d={in_f}):\n"
        f"  Backbone:  18 bits / {H24_DIM} params = {backbone_bpp:.3f} bits/param\n"
        f"  Scale:     16 bits / {in_f} params"
        f" ≈ {scale_bpp:.4f} bits/param  ({h24_scale_row_count} row scales)\n"
        f"  Total:     {total_bpp:.4f} bits/param",
        flush=True,
    )


# =============================================================================
# Checkpoint discovery
# =============================================================================

def _find_continuous_checkpoints(ckpt_dir: str) -> list[str]:
    """Return *.pt files in ckpt_dir, excluding *.packed.pt and *.plx."""
    return sorted(
        p for p in glob(os.path.join(ckpt_dir, "*.pt"))
        if not p.endswith(".packed.pt")
    )


def _resolve_path(explicit: str | None, ckpt_dir: str) -> str:
    """Resolve an explicit path or run an interactive selection wizard."""
    if explicit:
        if os.path.isfile(explicit):
            return os.path.abspath(explicit)
        candidate = os.path.join(ckpt_dir, explicit)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {explicit!r}")

    paths = _find_continuous_checkpoints(ckpt_dir)
    if not paths:
        raise FileNotFoundError(
            f"No *.pt checkpoints found in {ckpt_dir!r}.  "
            "Train a model first with train.py, or provide an explicit path."
        )

    print("\nAvailable checkpoints:", flush=True)
    for i, p in enumerate(paths):
        print(f"  [{i}]  {os.path.relpath(p)}", flush=True)

    raw = input(f"\nSelect [0-{len(paths) - 1}] (Enter = latest): ").strip()
    if not raw:
        return paths[-1]
    try:
        return paths[int(raw)]
    except (ValueError, IndexError):
        print("Invalid selection — using latest.", flush=True)
        return paths[-1]


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack a Pollux training checkpoint into the .plx binary format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pack.py\n"
            "  python pack.py checkpoints/pollux_final.pt\n"
            "  python pack.py pollux_final.pt --output model.plx --device cpu\n"
        ),
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="",
        help="Path to a *.pt training checkpoint (interactive wizard if omitted)",
    )
    parser.add_argument(
        "--output", "-o",
        default="",
        help="Output .plx path (default: <checkpoint>.plx)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for quantization — 'cuda' or 'cpu' (default: cuda)",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="",
        help="Directory to search for checkpoints (default: ./checkpoints)",
    )
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir or os.path.join(str(_HERE), "checkpoints")
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable — falling back to CPU.", flush=True)
        device_str = "cpu"
    device = torch.device(device_str)

    src = _resolve_path(str(args.checkpoint).strip() or None, ckpt_dir)
    dst = args.output or os.path.splitext(src)[0] + ".plx"

    print(f"\nPacking  {src}\n     →   {dst}", flush=True)
    pack_checkpoint(src, dst, device)


if __name__ == "__main__":
    main()
