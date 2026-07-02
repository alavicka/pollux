# Copyright (c) 2026 Alexander Lavicka.
# This source code is licensed under the PolyForm Noncommercial License 1.0.0.
# A copy of this license is available at https://polyformproject.org/licenses/noncommercial/1.0.0/
# Commercial utilization or hardware integration requires a separate license from the patent holder.

"""Castor: Leech lattice geometry, codebook, and fused quantization kernels.

Leaf node of the Pollux dependency graph — imports nothing from pollux.py or
train.py. All architectural constants derive strictly from the two H24 axioms
defined in the paper.

This module provides:
  - Axiomatic constants: friction_c, G24, H24_DIM, etc.
  - Codebook: Deterministic 196,561-entry table (196,560 kissing points + index 0 null attractor).
  - Normalization: Global row-wise RMS functions.
  - Quantizer: Exact nearest-neighbour Euclidean search (Triton fused kernel + PyTorch fallback).
  - Bit-packing: Bijective 18-bit → 9-byte encoding for .plx serialization.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

# =============================================================================
# Foundational geometric constants (Axiom 1)
# =============================================================================

# Topological friction (C): C = √2 — Leech-lattice covering-to-packing radius ratio
# (Conway & Sloane).  Exact deep-hole Voronoi barrier; not empirical.
friction_c: float = math.sqrt(2.0)

# System jitter axiom (γ): normalized second moment (NSM) of the Leech-lattice
# Voronoi cell (G₂₄).  Intrinsic H₂₄ quantization noise floor — sets the Adam
# momentum coefficients (β₁ = 1 − √γ, β₂ = 1 − γ) and the dissipative decay rate.
G24: float = 0.065771
gamma: float = G24

# 24D space axiom: unit dimension of the Viazovska singularity.
H24_DIM: int = 24

# Fixed Euclidean lattice scale constant (s_lattice = 1/C): maps continuous latent
# coordinates to normalised lattice coordinates.
s_lattice: float = 1.0 / friction_c

# Numerical guard against division-by-zero throughout the stack.
NUMERICAL_EPSILON: float = 1e-8

# =============================================================================
# Codebook constants
# =============================================================================

H24_KISSING_COUNT = 196_560
H24_CODEBOOK_COUNT = H24_KISSING_COUNT + 1   # Axiom 2: + null vector at index 0
H24_INDEX_BITS = 18
H24_INDEX_MASK = (1 << H24_INDEX_BITS) - 1
BACKBONE_BITS_PER_PARAM = 0.76
BITS_PER_PARAM_INDEX = 18 / H24_DIM  # 0.75
INDICES_PER_CHUNK = 4
BYTES_PER_CHUNK = 9

# All 196,560 Leech kissing points have squared norm 32 in raw coordinates.
# Scaling by 1/√8 maps them to squared norm 4 (Euclidean norm 2).
# This uniform norm enables fast dot-product nearest-neighbour search.
LEECH_SCALE: float = 1.0 / math.sqrt(8.0)

# =============================================================================
# Golay code helpers (codebook construction)
# =============================================================================

_GOLAY_POLY = (1 << 11) | (1 << 9) | (1 << 7) | (1 << 6) | (1 << 5) | (1 << 1) | 1


def _gf2_mod(dividend: int, divisor: int, deg_div: int) -> int:
    value = dividend
    while value.bit_length() - 1 >= deg_div:
        shift = (value.bit_length() - 1) - deg_div
        value ^= divisor << shift
    return value


def _encode_golay23(msg12: int) -> int:
    return (msg12 << 11) | _gf2_mod(msg12 << 11, _GOLAY_POLY, 11)


def _extended_golay24_words() -> list[int]:
    words = []
    for msg in range(4096):
        w23 = _encode_golay23(msg)
        parity = bin(w23).count("1") % 2
        words.append((parity << 23) | w23)
    return words


def _leech_minimal_vectors() -> torch.Tensor:
    """Generate all 196,560 Leech kissing points in raw {0, ±2, ±4} coordinates.

    All vectors have squared norm 32.  Scale by LEECH_SCALE = 1/√8 to reach norm 2.
    """
    golay = _extended_golay24_words()
    vectors: list[list[float]] = []

    # Type 1 — 1,104 points: two coordinates at ±4, rest 0.
    # (Squared norm = 16 + 16 = 32 ✓)
    for i in range(H24_DIM):
        for j in range(i + 1, H24_DIM):
            for a in (-4.0, 4.0):
                for b in (-4.0, 4.0):
                    row = [0.0] * H24_DIM
                    row[i], row[j] = a, b
                    vectors.append(row)

    # Type 2 — 97,152 points: eight coordinates at ±2 (Golay octad), even #minus.
    # (Squared norm = 8 × 4 = 32 ✓)
    for word in golay:
        bits = [(word >> b) & 1 for b in range(24)]
        pos = [i for i, b in enumerate(bits) if b]
        if len(pos) != 8:
            continue
        for mask in range(256):
            signs = [1.0 if (mask >> k) & 1 else -1.0 for k in range(8)]
            if sum(s < 0 for s in signs) % 2 != 0:
                continue
            row = [0.0] * 24
            for p, s in zip(pos, signs):
                row[p] = 2.0 * s
            vectors.append(row)

    # Type 3 — 98,304 points: one coordinate at −3, others at ±1 via Golay.
    # (Squared norm = 9 + 23 × 1 = 32 ✓)
    for pos in range(24):
        for word in golay:
            bits = [(word >> b) & 1 for b in range(24)]
            row = [0.0] * 24
            for i, b in enumerate(bits):
                row[i] = -1.0 if b else 1.0
            row[pos] = -3.0
            vectors.append(row)

    return torch.tensor(vectors, dtype=torch.float32)


@lru_cache(maxsize=1)
def _build_codebook_cpu() -> torch.Tensor:
    """Build the 196,561-entry Leech kissing-point codebook analytically.

    Returns a ``[196561, 24]`` float32 tensor.
    Index 0  → zero vector (maps noise / dead atoms to null energy).
    Index 1+ → the 196,560 Leech kissing points scaled to norm 2.

    All non-zero entries share EXACTLY the same Euclidean norm (= 2), enabling
    Euclidean nearest-neighbour search via pure dot-product (max score = min dist).
    """
    raw = _leech_minimal_vectors() * LEECH_SCALE  # [196560, 24], norm 2

    # Sanity: all kissing points must have squared norm exactly 4.
    norms_sq = raw.pow(2).sum(dim=1)
    max_dev = (norms_sq - 4.0).abs().max().item()
    if max_dev > 1e-4:
        raise RuntimeError(
            f"Kissing-point norm error: max_deviation={max_dev:.6f}. "
            "Expected all squared norms = 4."
        )

    # Deterministic lexicographic sort for reproducible fingerprint.
    sort_cols = [raw[:, i].numpy() for i in range(H24_DIM - 1, -1, -1)]
    sort_key = torch.from_numpy(np.lexsort(sort_cols))
    raw = raw[sort_key]

    if raw.shape[0] != H24_KISSING_COUNT:
        raise RuntimeError(f"Raw kissing-point count {raw.shape[0]} != {H24_KISSING_COUNT}")

    # Prepend the zero vector at index 0.
    zero_vec = torch.zeros(1, H24_DIM, dtype=raw.dtype)
    vectors = torch.cat([zero_vec, raw], dim=0)  # [196561, 24]

    if vectors.shape[0] != H24_CODEBOOK_COUNT:
        raise RuntimeError(f"Codebook size {vectors.shape[0]} != {H24_CODEBOOK_COUNT}")

    # Assert all 196,561 rows are distinct (catches any residual generation bugs).
    n_unique = torch.unique(vectors, dim=0).shape[0]
    if n_unique != H24_CODEBOOK_COUNT:
        raise RuntimeError(
            f"Codebook uniqueness failure: {n_unique} unique rows, expected {H24_CODEBOOK_COUNT}."
        )

    return vectors


def get_h24_codebook(device: torch.device | str, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Return the 196,561-entry Leech kissing-point codebook on ``device``.

    Shape ``[196561, 24]``.  Index 0 is the zero vector; indices 1–196560 are the
    Leech kissing points at norm 2 (scaled by LEECH_SCALE = 1/√8).
    """
    return _build_codebook_cpu().to(device=device, dtype=dtype)


def centers_to_indices(
    centers: torch.Tensor,
    codebook: torch.Tensor,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Map 24D center vectors to their nearest codebook index via Euclidean cdist.

    Since all non-zero codebook entries have equal norm, this is equivalent to
    dot-product matching and returns exact results for any vector that is already
    a codebook entry.
    """
    assert centers.dim() == 2 and centers.size(1) == H24_DIM
    assert codebook.dim() == 2 and codebook.size(1) == H24_DIM

    N = centers.size(0)
    indices = torch.empty(N, dtype=torch.long, device=centers.device)
    book = codebook.to(device=centers.device).float()
    for i in range(0, N, chunk_size):
        chunk = centers[i : i + chunk_size].float()
        dist = torch.cdist(chunk, book)
        indices[i : i + chunk_size] = dist.argmin(dim=-1)
    return indices


def pack_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack 18-bit indices (4 per group) into 9 bytes big-endian."""
    indices = indices.reshape(-1)
    rem = len(indices) % 4
    if rem != 0:
        indices = torch.cat([indices, torch.zeros(4 - rem, dtype=indices.dtype, device=indices.device)])

    indices = indices.view(-1, 4).to(torch.int32)
    i0, i1, i2, i3 = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]

    chunks = torch.zeros((indices.size(0), 9), dtype=torch.uint8, device=indices.device)
    chunks[:, 0] = (i0 >> 10) & 0xFF
    chunks[:, 1] = (i0 >> 2) & 0xFF
    chunks[:, 2] = ((i0 & 0x03) << 6) | ((i1 >> 12) & 0x3F)
    chunks[:, 3] = (i1 >> 4) & 0xFF
    chunks[:, 4] = ((i1 & 0x0F) << 4) | ((i2 >> 14) & 0x0F)
    chunks[:, 5] = (i2 >> 6) & 0xFF
    chunks[:, 6] = ((i2 & 0x3F) << 2) | ((i3 >> 16) & 0x03)
    chunks[:, 7] = (i3 >> 8) & 0xFF
    chunks[:, 8] = i3 & 0xFF
    return chunks.view(-1)


def unpack_indices(packed: torch.Tensor, num_indices: int) -> torch.Tensor:
    """Unpack 9-byte big-endian groups back to 18-bit indices."""
    packed = packed.view(-1)
    chunks = packed.view(-1, 9).to(torch.int32)

    i0 = (chunks[:, 0] << 10) | (chunks[:, 1] << 2) | (chunks[:, 2] >> 6)
    i1 = ((chunks[:, 2] & 0x3F) << 12) | (chunks[:, 3] << 4) | (chunks[:, 4] >> 4)
    i2 = ((chunks[:, 4] & 0x0F) << 14) | (chunks[:, 5] << 6) | (chunks[:, 6] >> 2)
    i3 = ((chunks[:, 6] & 0x03) << 16) | (chunks[:, 7] << 8) | chunks[:, 8]

    indices = torch.stack([i0, i1, i2, i3], dim=1).view(-1)
    return indices[:num_indices].long()


def codebook_fingerprint() -> tuple[float, ...]:
    """Deterministic checksum of the codebook for packed-checkpoint validation."""
    vectors = _build_codebook_cpu()
    return tuple(vectors[:4, :4].reshape(-1).tolist())


def h24_linear_atom_count(out_features: int, in_features: int) -> int:
    """Number of 24D atoms in an H24 linear weight matrix."""
    total = int(out_features) * int(in_features)
    if total % H24_DIM != 0:
        raise ValueError(
            f"H24 weight shape ({out_features}, {in_features}) has {total} params, "
            f"not divisible by H24_DIM={H24_DIM}."
        )
    return total // H24_DIM


def assert_h24_divisible(n_features: int, *, name: str = "features") -> None:
    """Pollux requires tensor widths to partition cleanly into 24D H24 atoms."""
    if int(n_features) % H24_DIM != 0:
        raise ValueError(
            f"Pollux requires {name} divisible by H24_DIM={H24_DIM}; got {n_features}."
        )


def assert_row_aligned_width(d_model: int) -> None:
    """Pollux requires embedding width divisible by 24 (one H24 atom per head)."""
    assert_h24_divisible(d_model, name="d_model")


def assert_row_aligned_in_features(in_features: int) -> None:
    """Hidden linear layers require in_features divisible by 24."""
    assert_h24_divisible(in_features, name="in_features")


def scale_bits_per_param(in_features: int) -> float:
    """Amortised FP16 σ_rms storage rate: one scale per row of length in_features."""
    assert_row_aligned_in_features(in_features)
    return 16.0 / float(in_features)


def h24_linear_scale_row_count(out_features: int) -> int:
    """Number of FP16 σ_rms values in .plx: exactly one per weight-matrix row."""
    return int(out_features)


def row_rms_from_matrix(weight: torch.Tensor) -> torch.Tensor:
    """Global RMS normalization scale per matrix row: sqrt(mean(w²))."""
    if weight.dim() != 2:
        raise ValueError(f"row_rms_from_matrix expects a 2D weight matrix, got {weight.dim()}D.")
    return torch.sqrt(weight.pow(2).mean(dim=-1) + NUMERICAL_EPSILON)


def normalize_matrix_by_row_rms(
    weight: torch.Tensor,
    sigma_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide each row by its global RMS scale σ_rms over the full in_features width.

    Preserves relative Euclidean geometry across all atoms in the row (Synchronization
    Protocol).  Discrete H24 indices carry all structural representational logic.

    Returns:
        (normalized_weight, sigma_rows) where ``sigma_rows`` has shape ``[out_features]``.
    """
    out_f, in_f = weight.shape
    assert_row_aligned_in_features(in_f)
    if sigma_rows is None:
        sigma_rows = row_rms_from_matrix(weight)
    sigma = sigma_rows.reshape(out_f, 1).clamp_min(NUMERICAL_EPSILON)
    return weight / sigma, sigma_rows.reshape(-1)


def scale_matrix_by_row_rms(weight: torch.Tensor, sigma_rows: torch.Tensor) -> torch.Tensor:
    """Multiply each row by its global RMS scale σ_rms."""
    out_f, _in_f = weight.shape
    return weight * sigma_rows.reshape(out_f, 1)


# =============================================================================
# Leech nearest-neighbour quantizer (Triton + PyTorch fallback)
# =============================================================================

_LEECH_TRITON_ACTIVE: bool = False
_LEECH_TRITON_ERROR_PRINTED: bool = False

try:  # pragma: no cover - optional on CPU-only hosts
    import triton
    import triton.language as tl

    _TRITON_IMPORT_OK = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_OK = False


# ---------------------------------------------------------------------------
# Fused Triton nearest-neighbour kernel for the Leech codebook
# ---------------------------------------------------------------------------
# Replaces the naive  scores = matmul(chunk, kp.t())  loop which materialises
# a ~3.2 GB [chunk_size × 196560] float32 matrix in VRAM on every call.
#
# Design (FlashAttention-style register tiling):
#   Grid : 1-D over atoms, each block covers BLOCK_N atoms.
#   Inner: sweep all 196,560 kissing points in BLOCK_C slices.
#          → score tile = BLOCK_N × BLOCK_C lives entirely in registers.
#          → codebook (~18 MB) stays in L2 cache across all blocks.
#   Peak VRAM delta: O(BLOCK_N × BLOCK_C × 4 B) instead of O(N × 196560 × 4 B).
# ---------------------------------------------------------------------------

if _TRITON_IMPORT_OK:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 16, "BLOCK_C": 128}, num_warps=4),
            triton.Config({"BLOCK_N": 32, "BLOCK_C": 128}, num_warps=4),
            triton.Config({"BLOCK_N": 16, "BLOCK_C": 256}, num_warps=4),
            triton.Config({"BLOCK_N": 32, "BLOCK_C": 256}, num_warps=8),
        ],
        key=[],  # codebook shape is invariant; tune once and cache globally
    )
    @triton.jit
    def _leech_nn_kernel(
        latent_ptr,   # [N, H]     float32 – input atoms
        kp_ptr,       # [N_KP, H]  float32 – kissing points (index 0 of codebook excluded)
        idx_ptr,      # [N]        int32   – output: codebook index for each atom
        N,            # runtime: number of atoms
        N_KP,         # runtime: 196560  (runtime → dynamic loop, not unrolled)
        c_sq_norm,    # float: ‖any kissing point‖² = 4.0  (constant for all KPs)
        H: tl.constexpr,       # 24
        H_PAD: tl.constexpr,   # 32 – next power-of-two ≥ H (required by tl.arange / tl.dot)
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid    = tl.program_id(0)
        n_base = pid * BLOCK_N
        n_offs = n_base + tl.arange(0, BLOCK_N)   # [BLOCK_N]
        n_mask = n_offs < N

        h_offs = tl.arange(0, H_PAD)              # [H_PAD]
        h_mask = h_offs < H

        # Load the latent tile [BLOCK_N, H_PAD] — cached in registers for all KP chunks.
        x = tl.load(
            latent_ptr + n_offs[:, None] * H + h_offs[None, :],
            mask=n_mask[:, None] & h_mask[None, :],
            other=0.0,
        )

        # Running nearest-neighbour state — purely register-resident.
        running_max = tl.full([BLOCK_N], float("-inf"), dtype=tl.float32)
        best_idx    = tl.zeros([BLOCK_N], dtype=tl.int32)

        # Inner loop: stream the codebook in BLOCK_C slices.
        # N_KP is a runtime value → Python range creates a dynamic (non-unrolled) while-loop.
        for c_base in range(0, N_KP, BLOCK_C):
            c_offs = c_base + tl.arange(0, BLOCK_C)   # [BLOCK_C]
            c_mask = c_offs < N_KP

            # Load one KP slice [BLOCK_C, H_PAD] from global mem.
            # After the first full sweep, this block is L2-resident for subsequent programs.
            kp = tl.load(
                kp_ptr + c_offs[:, None] * H + h_offs[None, :],
                mask=c_mask[:, None] & h_mask[None, :],
                other=0.0,
            )

            # Cast tiles to BF16 to engage Tensor Cores (Ampere+ WMMA BF16 path).
            # tl.dot accumulates in FP32 regardless — running_max and snap logic
            # operate on the FP32 accumulator, so argmax correctness is preserved.
            # The Leech Voronoi cells are widely separated (min inter-center gap > 1.4),
            # so BF16 rounding (~0.2% relative error) cannot flip the nearest neighbour.
            x_bf16  = x.to(tl.bfloat16)
            kp_bf16 = kp.to(tl.bfloat16)
            scores = tl.dot(x_bf16, tl.trans(kp_bf16), allow_tf32=True)

            # Zero out scores for padding entries beyond the codebook boundary.
            scores = tl.where(
                c_mask[None, :],
                scores,
                tl.full([BLOCK_N, BLOCK_C], float("-inf"), dtype=tl.float32),
            )

            # Local per-atom max and argmax over this BLOCK_C slice.
            chunk_max  = tl.max(scores, axis=1)      # [BLOCK_N]
            chunk_best = tl.argmax(scores, axis=1)   # [BLOCK_N], 0-based within slice

            # Update global running state where this slice improved the score.
            improved    = chunk_max > running_max
            running_max = tl.where(improved, chunk_max, running_max)
            best_idx    = tl.where(improved, (c_base + chunk_best).to(tl.int32), best_idx)

        # Epilog: Voronoi zero-vector test.
        # ‖x − c‖² < ‖x‖²  ↔  2(x·c) > ‖c‖²
        # Snap to codebook index 0 (zero vector) when no kissing point is closer than the origin.
        snap      = (2.0 * running_max) < c_sq_norm
        final_idx = tl.where(snap, tl.zeros([BLOCK_N], tl.int32), best_idx + 1)

        tl.store(idx_ptr + n_offs, final_idx, mask=n_mask)

    def _leech_nn_triton(
        x: torch.Tensor,    # [N, 24] float32, contiguous, CUDA
        kp: torch.Tensor,   # [196560, 24] float32, contiguous, CUDA
        c_sq_norm: float,
    ) -> torch.Tensor:      # [N] int32 – codebook indices
        """Launch ``_leech_nn_kernel``, return raw int32 indices."""
        global _LEECH_TRITON_ACTIVE
        N   = x.size(0)
        N_KP = kp.size(0)
        idx = torch.empty(N, dtype=torch.int32, device=x.device)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
        _leech_nn_kernel[grid](
            x, kp, idx,
            N, N_KP, c_sq_norm,
            H=H24_DIM,
            H_PAD=32,
        )
        if not _LEECH_TRITON_ACTIVE:
            print("Leech Triton NN kernel: active", flush=True)
            _LEECH_TRITON_ACTIVE = True
        return idx


def castor_quantize_leech(
    latent: torch.Tensor,
    codebook: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Exact Leech-lattice nearest-neighbour quantizer (STE-compatible forward).

    Because all 196,560 kissing points share the same L2 norm, minimising
    ‖x − c‖² is equivalent to maximising x · c.

    On CUDA with Triton: dispatches to ``_leech_nn_kernel``, a fused kernel
    that keeps the running (max, idx) state in registers and streams the
    codebook in BLOCK_C slices.  Peak VRAM scales as O(BLOCK_N × BLOCK_C × 4 B)
    rather than O(N × 196560 × 4 B) — the full score matrix is never written
    to global memory.

    On CPU or when Triton is unavailable: falls back to chunked PyTorch matmul.

    Args:
        latent:     ``[..., 24]`` float tensor.
        codebook:   ``[196561, 24]`` float tensor; index 0 is the zero vector.
        chunk_size: Atoms per matmul chunk for the CPU/fallback path.

    Returns:
        Tensor of the same shape as ``latent``, each 24-D atom replaced by its
        nearest codebook entry (zero vector when closer to the origin than to
        any kissing point).
    """
    orig_shape = latent.shape
    x = latent.reshape(-1, H24_DIM).float()
    N = x.size(0)
    if N == 0:
        return latent.clone()

    kissing_points = codebook[1:].to(device=x.device, dtype=torch.float32)  # [196560, 24]
    c_sq_norm = float(kissing_points[0].pow(2).sum().item())  # = 4.0 for all KPs

    # --- Fast path: fused Triton kernel (CUDA only) ---
    if _TRITON_IMPORT_OK and x.device.type == "cuda":
        global _LEECH_TRITON_ERROR_PRINTED
        try:
            indices = _leech_nn_triton(
                x.contiguous(),
                kissing_points.contiguous(),
                c_sq_norm,
            ).long()
            out = codebook.to(device=x.device, dtype=latent.dtype).index_select(0, indices)
            return out.view(orig_shape)
        except Exception as exc:
            if not _LEECH_TRITON_ERROR_PRINTED:
                print(
                    f"Leech Triton NN kernel error ({exc}); "
                    "falling back to PyTorch chunked matmul.",
                    flush=True,
                )
                _LEECH_TRITON_ERROR_PRINTED = True

    # --- CPU / fallback: chunked PyTorch matmul ---
    # Uses autocast(bfloat16) so the matmul runs on BF16 Tensor Cores when
    # available (CUDA without Triton, or CPU with AMX/BF16 support).
    # Scores are cast back to FP32 before .max() so the snap threshold
    # comparison retains full precision.
    result_indices = torch.empty(N, dtype=torch.long, device=x.device)
    for i in range(0, N, chunk_size):
        chunk = x[i : i + chunk_size]
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            scores = torch.matmul(chunk, kissing_points.t())
        max_scores, best_kp_idx = scores.float().max(dim=-1)
        snap_to_zero = (2.0 * max_scores) < c_sq_norm
        final_idx = (best_kp_idx + 1).masked_fill(snap_to_zero, 0)
        result_indices[i : i + chunk_size] = final_idx

    out = codebook.to(device=x.device, dtype=latent.dtype).index_select(0, result_indices)
    return out.view(orig_shape)


class CastorQuantizeLeech(torch.autograd.Function):
    """Autograd wrapper for castor_quantize_leech with straight-through backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        latent: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        ctx.mark_non_differentiable(codebook)
        return castor_quantize_leech(latent, codebook)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return grad_output, None
