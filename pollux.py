# Copyright (c) 2026 Alexander Lavicka.
# This source code is licensed under the PolyForm Noncommercial License 1.0.0.
# A copy of this license is available at https://polyformproject.org/licenses/noncommercial/1.0.0/
# Commercial utilization or hardware integration requires a separate license from the patent holder.

"""Pollux: zero-continuous-weight transformer backbone and thermodynamic estimator.

Depends only on castor.py (the axiom layer). Implements the full native H24
Leech-lattice quantization pipeline: continuous pre-weights (optimiser state
only) → Voronoi projection → fully discrete structural centres on the
backbone → 0.76-bit packed inference weights.

This module provides:
  - Thermodynamic constants: DATASET_NOISE_FLOOR, GEOMETRIC_SLACK, KINEMATIC_LIMIT, CRITICAL_DIM.
  - State Management: PolluxState for zero-overhead thermodynamic tracking.
  - Architecture: PolluxModel, PolluxH24Linear (training), PackedH24Linear (inference), and peripheral layers.
  - Optimizer: pollux_step (parameter-free thermodynamic estimator).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from castor import (
    BACKBONE_BITS_PER_PARAM,
    G24,
    H24_DIM,
    friction_c,
    gamma,
    s_lattice,
    NUMERICAL_EPSILON,
    assert_row_aligned_in_features,
    assert_row_aligned_width,
    castor_quantize_leech,
    codebook_fingerprint,
    get_h24_codebook,
    h24_linear_atom_count,
    h24_linear_scale_row_count,
    normalize_matrix_by_row_rms,
    scale_matrix_by_row_rms,
    unpack_indices,
)

C: float = float(friction_c)  # Axiom 1; topological friction (Leech covering ratio)
# gamma (= G24) imported from castor — NSM of the Leech Voronoi cell.

# Irreducible cross-entropy floor of the training corpus (empirical material
# property; see paper §3.4.1 — FP16 continuous-weight baseline on FineWeb-Edu).
DATASET_NOISE_FLOOR: float = 3.2

# Deep-hole slack above unit packing radius (Axiom 1): C − 1.
GEOMETRIC_SLACK: float = C - 1.0

# Per-24D-atom kinematic speed cap: half the geometric slack, in lattice units.
KINEMATIC_LIMIT: float = GEOMETRIC_SLACK / 2.0

# Width-inertia equilibrium d* = (24·C)² = 1152 for C = √2 — width at which η_d = 1.
CRITICAL_DIM: float = (float(H24_DIM) ** 2) * (C ** 2)

_H24_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}

# =============================================================================
# H24 basis helpers
# =============================================================================


def _generate_h24_basis_once() -> torch.Tensor:
    """Construct the 24×24 H24 basis from the extended binary Golay structure."""
    q = 11
    residues = {(k * k) % q for k in range(1, q)}
    def _chi(x: int) -> int:
        xm = x % q
        if xm == 0:
            return 0
        return 1 if xm in residues else -1
    j = torch.empty((q, q), dtype=torch.float32)
    for r in range(q):
        for c in range(q):
            j[r, c] = float(_chi(c - r))
    cmat = torch.empty((q + 1, q + 1), dtype=torch.float32)
    cmat[0, 0] = 0.0
    cmat[0, 1:] = 1.0
    cmat[1:, 0] = -1.0
    cmat[1:, 1:] = j
    h12 = torch.eye(q + 1, dtype=torch.float32) + cmat
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32)
    return torch.kron(h2, h12) / math.sqrt(float(H24_DIM))


H24 = _generate_h24_basis_once()


def _get_h24(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the cached H24 basis on the requested device and dtype."""
    key = (str(device), dtype)
    cached = _H24_CACHE.get(key)
    if cached is not None:
        return cached
    h24 = H24.to(device=device, dtype=dtype).contiguous()
    _H24_CACHE[key] = h24
    return h24


def _z(device: torch.device, dtype: torch.dtype, value: float) -> torch.Tensor:
    """Scalar zero-dimensional accumulator tensor on device."""
    return torch.as_tensor(value, device=device, dtype=dtype).reshape(())


def _is_h24_tensor(p: torch.Tensor) -> bool:
    """True when p is a quantizable weight matrix (≥2D, element count divisible by 24)."""
    return p.dim() >= 2 and p.numel() > 0 and (p.numel() % H24_DIM == 0)


def h24_basis_fingerprint() -> tuple[float, ...]:
    """Deterministic checksum of the H24 basis for reproducibility audits."""
    return tuple(H24.flatten()[:8].tolist())


def entropy_ceiling(vocab_size: float) -> float:
    """Maximum cross-entropy (uniform prior) for a vocabulary of given size."""
    return math.log(float(vocab_size))


# =============================================================================
# PolluxState — thermodynamic and update-dynamics field state
# =============================================================================

_PHYSICS_SCALAR_NAMES = (
    "step",
    "grad_accum_steps",
    "n_embd",
    "vocab_size",
    "current_loss",
    "macro_heat",
    "adam_force",
)


class PolluxState:
    """Persistent thermodynamic and update-dynamics state bound to a Pollux model.

    Scalar fields remain on device as 0-D tensors until explicitly read for
    logging. ``n_embd`` and ``vocab_size`` feed the width-dependent representational
    stability and entropy-ceiling calculations in pollux_step.
    """

    def __init__(self, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> None:
        dev = device or torch.device("cpu")
        self.device = dev
        self.dtype = dtype
        self.step = torch.zeros((), device=dev, dtype=torch.long)
        self.grad_accum_steps = torch.ones((), device=dev, dtype=torch.long)
        self.n_embd = _z(dev, dtype, float(PolluxConfig.n_embd))
        self.vocab_size = _z(dev, dtype, float(PolluxConfig.vocab_size))
        self.current_loss = _z(dev, dtype, 0.0)
        self.macro_heat = _z(dev, dtype, 0.0)
        self.adam_force = _z(dev, dtype, 0.0)
        self.m_buffers: dict[str, torch.Tensor] = {}
        self.v_buffers: dict[str, torch.Tensor] = {}
        self.continuous_pre_weights: dict[str, torch.Tensor] = {}
        self.tracked_params: dict[str, torch.Tensor] = {}

    def bind_model_buffers(self, model: torch.nn.Module) -> None:
        """Attach trainable parameters and sync architecture scalars from the model config."""
        core = getattr(model, "_orig_mod", model)
        cfg = getattr(core, "cfg", None)
        if cfg is not None:
            self.n_embd.copy_(
                torch.as_tensor(float(int(getattr(cfg, "n_embd", PolluxConfig.n_embd))), device=self.device, dtype=self.dtype)
            )
            self.vocab_size.copy_(
                torch.as_tensor(float(int(getattr(cfg, "vocab_size", PolluxConfig.vocab_size))), device=self.device, dtype=self.dtype)
            )
        self.tracked_params = {
            name: p
            for name, p in core.named_parameters()
            if p.requires_grad and p.dtype.is_floating_point
        }

    def kinetic_buffers(self, name: str, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (m, v) Adam moment buffers, lazily allocated per parameter."""
        m = self.m_buffers.get(name)
        if m is None or m.shape != p.shape or m.device != p.device or m.dtype != p.dtype:
            m = torch.zeros_like(p)
            self.m_buffers[name] = m
        v = self.v_buffers.get(name)
        if v is None or v.shape != p.shape or v.device != p.device or v.dtype != p.dtype:
            v = torch.zeros_like(p)
            self.v_buffers[name] = v
        return m, v

    def latent_buffer(self, name: str, p: torch.Tensor) -> torch.Tensor:
        """Return continuous pre-weights for parameter ``name``, initialized from ``p.data``."""
        latent = self.continuous_pre_weights.get(name)
        if (
            latent is None
            or latent.shape != p.shape
            or latent.device != p.device
            or latent.dtype != p.dtype
        ):
            latent = p.data.detach().clone()
            self.continuous_pre_weights[name] = latent
        return latent

    def state_dict(self) -> dict[str, Any]:
        """Serialize physics scalars and per-parameter Adam-momentum/pre-weight buffers."""
        tensors = {name: getattr(self, name).detach().clone() for name in _PHYSICS_SCALAR_NAMES}
        return {
            "tensors": tensors,
            "m_buffers": {k: v.detach().clone() for k, v in self.m_buffers.items()},
            "v_buffers": {k: v.detach().clone() for k, v in self.v_buffers.items()},
            "continuous_pre_weights": {
                k: v.detach().clone() for k, v in self.continuous_pre_weights.items()
            },
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore physics state from a serialized checkpoint payload."""
        if not isinstance(payload, dict):
            raise ValueError("physics_state_dict must be a mapping with tensor payloads.")

        # Legacy checkpoint migration: old keys and removed observables are accepted
        # but no longer required on load.
        payload = dict(payload)
        if "continuous_pre_weights" not in payload and "latent_plasma_buffers" in payload:
            payload["continuous_pre_weights"] = payload["latent_plasma_buffers"]

        tensors = payload.get("tensors", {})
        for name in _PHYSICS_SCALAR_NAMES:
            if name in tensors:
                getattr(self, name).copy_(tensors[name].to(device=self.device))
        for key, buf in payload.get("m_buffers", {}).items():
            self.m_buffers[key] = buf.to(device=self.device)
        for key, buf in payload.get("v_buffers", {}).items():
            self.v_buffers[key] = buf.to(device=self.device)
        for key, buf in payload.get("continuous_pre_weights", {}).items():
            self.continuous_pre_weights[key] = buf.to(device=self.device)

    def metrics_for_wandb(self) -> dict[str, float]:
        """Export scalar observables for experiment logging."""
        return {
            "loss/current": float(self.current_loss.item()),
            "thermo/macro_heat": float(self.macro_heat.item()),
            "kinetic/adam_force": float(self.adam_force.item()),
        }


# =============================================================================
# Architecture
# =============================================================================

_CONFIG_FIELDS = (
    "n_layers",
    "n_heads",
    "d_head",
    "n_embd",
    "tokenizer_vocab_size",
    "vocab_size",
    "batch_size",
    "seq_len",
    "grad_accum_steps",
    "use_compile",
    "use_gradient_checkpointing",
    "target_tokens",
    "tokenizer_dir",
)

_CONFIG_ALIASES = {
    "d_model": "n_embd",
    "n_layer": "n_layers",
    "n_head": "n_heads",
}

class PolluxConfig:
    """Architecture and training hyperparameters for the Pollux field."""

    n_layers = 18
    n_heads = 80
    d_head = 24
    n_embd = 1920  # must be divisible by 24 (H24 atom tiling)
    tokenizer_vocab_size = 50257
    vocab_size = 50688
    batch_size = 2
    seq_len = 1024
    grad_accum_steps = 128
    use_compile = False
    use_gradient_checkpointing = False
    target_tokens = 9_953_989_333
    tokenizer_dir = ""

    def __init__(self, **overrides: Any) -> None:
        for name in _CONFIG_FIELDS:
            if name in overrides:
                setattr(self, name, overrides[name])

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for key, value in raw.items():
            target = _CONFIG_ALIASES.get(str(key), str(key))
            if target in _CONFIG_FIELDS:
                merged[target] = value
        return merged

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PolluxConfig:
        normalized = cls._normalize_dict(raw or {})
        values = {name: getattr(cls, name) for name in _CONFIG_FIELDS}
        values.update(normalized)
        cfg = cls(**values)
        cfg.validate_instance()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _CONFIG_FIELDS}

    @staticmethod
    def _validate_values(cfg: PolluxConfig) -> None:
        d_head = int(cfg.d_head)
        n_heads = int(cfg.n_heads)
        n_embd = int(cfg.n_embd)
        if d_head != H24_DIM:
            raise ValueError(
                f"d_head must equal H24_DIM ({H24_DIM}): each attention head is one "
                "24-dimensional Leech lattice atom."
            )
        if n_embd != (n_heads * d_head):
            raise ValueError(
                "n_embd must equal n_heads * d_head: embedding width is an integer "
                "multiple of the 24D packing unit."
            )
        assert_row_aligned_width(n_embd)
        vocab_size = int(cfg.vocab_size)
        tokenizer_vocab_size = int(cfg.tokenizer_vocab_size)
        if vocab_size % H24_DIM != 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be divisible by H24_DIM ({H24_DIM})."
            )
        if vocab_size < tokenizer_vocab_size:
            raise ValueError(
                f"vocab_size ({vocab_size}) must cover tokenizer_vocab_size "
                f"({tokenizer_vocab_size})."
            )

    @classmethod
    def validate(cls) -> None:
        cls._validate_values(cls())

    def validate_instance(self) -> None:
        PolluxConfig._validate_values(self)


def infer_config_from_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Recover PolluxConfig fields from a saved model state dictionary."""
    tok_emb = state_dict.get("tok_emb.weight")
    if tok_emb is None:
        return {}
    pos_emb = state_dict.get("pos_emb.weight")
    vocab_size = int(tok_emb.shape[0])
    n_embd = int(tok_emb.shape[1])
    seq_len = int(pos_emb.shape[0]) if pos_emb is not None else int(PolluxConfig.seq_len)
    n_layers = sum(
        1 for key in state_dict if key.startswith("blocks.") and key.endswith(".ln1.weight")
    ) or int(PolluxConfig.n_layers)
    d_head = H24_DIM
    n_heads = n_embd // d_head if d_head else int(PolluxConfig.n_heads)
    if vocab_size > 8192:
        tokenizer_vocab_size = min(50257, vocab_size)
    elif vocab_size > 4096:
        tokenizer_vocab_size = 4096
    else:
        tokenizer_vocab_size = vocab_size
    return {
        "vocab_size": vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "n_embd": n_embd,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_head": d_head,
        "seq_len": seq_len,
    }


def config_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    state_dict: dict[str, Any] | None = None,
) -> PolluxConfig:
    """Build PolluxConfig from an explicit config dict or inferred weight shapes."""
    raw = checkpoint.get("config")
    if isinstance(raw, dict) and raw:
        return PolluxConfig.from_dict(raw)
    inferred = infer_config_from_state_dict(state_dict or checkpoint.get("model_state_dict", {}))
    if inferred:
        return PolluxConfig.from_dict(inferred)
    raise ValueError(
        "Cannot reconstruct PolluxConfig: checkpoint contains neither a 'config' "
        "dict nor a recognisable 'model_state_dict'.  Provide a valid Pollux "
        "training checkpoint produced by train.py."
    )


class PolluxH24Linear(nn.Linear):
    """Zero-continuous-weight H24 linear map with unit-variance initialization.

    ``self.weight`` holds the materialised discrete H24 centres (indices +
    row-wise RMS scales) used by ``forward``.  Continuous pre-weights live in
    ``PolluxState.continuous_pre_weights`` and are updated in ``pollux_step``;
    after each step the observable ``p.data`` is overwritten with the
    re-quantised lattice projection.  Forward and backward never invoke the
    codebook — latent-parameter swapping only.

    ``vacuum_width`` is retained for API compatibility (equals ``n_embd`` for
    all backbone layers).
    """

    vacuum_width: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        vacuum_width: int,
    ) -> None:
        self.vacuum_width = int(vacuum_width)
        assert_row_aligned_in_features(in_features)
        super().__init__(in_features, out_features, bias=bias)

    def reset_parameters(self) -> None:
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


def _make_h24_linear(
    in_features: int,
    out_features: int,
    *,
    packed: bool,
    bias: bool = False,
    vacuum_width: int,
) -> nn.Module:
    if packed:
        return PackedH24Linear(in_features, out_features, bias=bias)
    return PolluxH24Linear(
        in_features, out_features, bias=bias, vacuum_width=vacuum_width
    )


class PackedH24Linear(nn.Module):
    """SRAM-fusion H24 linear layer: uint8 packed indices + FP16 σ_rms per row.

    At inference, materialize() expands indices to dense FP16 weights for
    standard PyTorch cuBLAS — see README \"Hardware & Inference Limitations\".
    One FP16 scale per weight-matrix row matches the global RMS projection.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        assert_row_aligned_in_features(in_features)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_atoms = h24_linear_atom_count(self.out_features, self.in_features)
        self.num_sigma_rows = h24_linear_scale_row_count(self.out_features)
        packed_len = ((self.num_atoms + 3) // 4) * 9
        self.register_buffer("packed_weights", torch.zeros(packed_len, dtype=torch.uint8))
        self.register_buffer("sigma_rows", torch.ones(self.num_sigma_rows, dtype=torch.float16))
        self._materialized = False
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

    def materialize(self) -> None:
        if self._materialized:
            return
        device = self.packed_weights.device
        dtype = self.sigma_rows.dtype

        indices = unpack_indices(self.packed_weights, self.num_atoms)
        atoms_euclidean = get_h24_codebook(device, dtype).index_select(0, indices)
        sigma_rows = self.sigma_rows.to(dtype=dtype)

        atoms = atoms_euclidean.view(self.out_features, self.in_features // H24_DIM, H24_DIM)
        # Reconstruct: w = kp · s_lattice · σ_rms  (one global scale per row)
        weight = (atoms * sigma_rows.view(-1, 1, 1) * s_lattice).reshape(
            self.out_features, self.in_features
        )
        self.weight = nn.Parameter(weight, requires_grad=False)

        del self.packed_weights
        del self.sigma_rows
        self._materialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


class PackedInt8Embedding(nn.Module):
    """Token embedding with INT8 weights and per-row FP16 operational scales μ."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.register_buffer("weight_int8", torch.zeros(num_embeddings, embedding_dim, dtype=torch.int8))
        self.register_buffer("mu_rows", torch.ones(num_embeddings, dtype=torch.float16))
        self._materialized = False

    def materialize(self) -> None:
        if self._materialized:
            return
        dtype = self.mu_rows.dtype
        # Reconstruct:  w = int8 · (μ / 127)  per row
        weight = self.weight_int8.to(dtype=dtype) * (self.mu_rows.unsqueeze(-1).to(dtype=dtype) / 127.0)
        self.weight = nn.Parameter(weight, requires_grad=False)
        del self.weight_int8
        del self.mu_rows
        self._materialized = True

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight.to(dtype=torch.float32))


class PackedInt8Linear(nn.Module):
    """Output projection with INT8 weights and per-row FP16 operational scales μ."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("mu_rows", torch.ones(out_features, dtype=torch.float16))
        self._materialized = False
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def materialize(self) -> None:
        if self._materialized:
            return
        dtype = self.mu_rows.dtype
        # Reconstruct:  w = int8 · (μ / 127)  per row
        weight = self.weight_int8.to(dtype=dtype) * (self.mu_rows.unsqueeze(-1).to(dtype=dtype) / 127.0)
        self.weight = nn.Parameter(weight, requires_grad=False)
        del self.weight_int8
        del self.mu_rows
        self._materialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


class RMSNorm(nn.Module):
    """Root-mean-square normalization with learnable per-dimension gains.

    Operates on the final dimension; 1D gain tensors are treated as semantic
    scalars in pollux_step (no width-dependent representational stability scaling).
    Gains manage residual-stream amplitude so H24 pre-weights encode directional
    geometry without fighting Landauer erasure.
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class PolluxCausalAttention(nn.Module):
    """Multi-head causal self-attention with H24 projections and Q/K normalization.

    Q and K are RMS-normalized per 24D head atom before dot-product attention,
    preventing softmax saturation in deep stacks without altering V or output paths.
    """

    def __init__(self, cfg: PolluxConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = int(cfg.n_heads)
        self.d_head = int(cfg.d_head)
        self.n_embd = int(cfg.n_embd)
        packed = bool(getattr(cfg, "packed_inference", False))
        self.q_proj = _make_h24_linear(
            self.n_embd, self.n_embd, packed=packed, bias=False, vacuum_width=self.n_embd
        )
        self.k_proj = _make_h24_linear(
            self.n_embd, self.n_embd, packed=packed, bias=False, vacuum_width=self.n_embd
        )
        self.v_proj = _make_h24_linear(
            self.n_embd, self.n_embd, packed=packed, bias=False, vacuum_width=self.n_embd
        )
        self.o_proj = _make_h24_linear(
            self.n_embd, self.n_embd, packed=packed, bias=False, vacuum_width=self.n_embd
        )
        # Per-head Q/K norms on the 24D Leech atom (d_head = H24_DIM).
        self.q_ln = RMSNorm(int(self.cfg.d_head))
        self.k_ln = RMSNorm(int(self.cfg.d_head))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.d_head)
        k = self.k_proj(x).view(b, t, self.n_heads, self.d_head)
        q = self.q_ln(q)
        k = self.k_ln(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = attn.transpose(1, 2).contiguous().view(b, t, c)
        return self.o_proj(y)


class PolluxMLP(nn.Module):
    """GELU feed-forward block with 4× expansion, H24-quantized projections."""

    def __init__(self, cfg: PolluxConfig) -> None:
        super().__init__()
        packed = bool(getattr(cfg, "packed_inference", False))
        hidden = int(cfg.n_embd) * 4
        n_embd = int(cfg.n_embd)
        self.fc1 = _make_h24_linear(
            n_embd, hidden, packed=packed, bias=False, vacuum_width=n_embd
        )
        self.fc2 = _make_h24_linear(
            hidden, n_embd, packed=packed, bias=False, vacuum_width=n_embd
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class PolluxBlock(nn.Module):
    """Pre-norm transformer block: attention and MLP residuals on the H24 field."""

    def __init__(self, cfg: PolluxConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln1 = RMSNorm(int(cfg.n_embd))
        self.attn = PolluxCausalAttention(cfg)
        self.ln2 = RMSNorm(int(cfg.n_embd))
        self.mlp = PolluxMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PolluxModel(nn.Module):
    """Full Pollux decoder: embeddings, H24 blocks, final norm, and LM head."""

    def __init__(self, cfg: PolluxConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else PolluxConfig()
        self.cfg.validate_instance()
        packed = bool(getattr(self.cfg, "packed_inference", False))
        std_init = 1.0 / (C * math.sqrt(int(self.cfg.n_embd)))

        if packed:
            self.tok_emb = PackedInt8Embedding(int(self.cfg.vocab_size), int(self.cfg.n_embd))
        else:
            self.tok_emb = nn.Embedding(int(self.cfg.vocab_size), int(self.cfg.n_embd))
            torch.nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std_init)

        self.pos_emb = nn.Embedding(int(self.cfg.seq_len), int(self.cfg.n_embd))
        torch.nn.init.normal_(self.pos_emb.weight, mean=0.0, std=std_init)
        self.blocks = nn.ModuleList([PolluxBlock(self.cfg) for _ in range(int(self.cfg.n_layers))])
        self.ln_f = RMSNorm(int(self.cfg.n_embd))
        if packed:
            self.head = PackedInt8Linear(int(self.cfg.n_embd), int(self.cfg.vocab_size), bias=False)
        else:
            self.head = nn.Linear(int(self.cfg.n_embd), int(self.cfg.vocab_size), bias=False)
            torch.nn.init.normal_(self.head.weight, mean=0.0, std=std_init)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        use_ckpt = bool(getattr(self.cfg, "use_gradient_checkpointing", False))
        for block in self.blocks:
            if use_ckpt:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @classmethod
    def from_packed_checkpoint(
        cls,
        file_path: str,
        device: torch.device,
        *,
        payload: dict[str, Any] | None = None,
    ) -> PolluxModel:
        """Load a SRAM-fusion packed checkpoint for low-VRAM inference."""
        if payload is None:
            payload = torch.load(file_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "pollux_packed_v4":
            raise ValueError(f"Unsupported packed format: {payload.get('format')!r}")

        ckpt_fp = payload.get("codebook_fingerprint")
        if ckpt_fp is not None and tuple(ckpt_fp) != codebook_fingerprint():
            raise ValueError(
                "H24 codebook fingerprint mismatch: packed checkpoint was built with a "
                "different kissing-point table than the current runtime."
            )

        cfg = PolluxConfig.from_dict(payload.get("config"))
        cfg.packed_inference = True
        model = cls(cfg)
        state = payload["state_dict"]

        load_state: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            if key.endswith(".packed_weights"):
                load_state[key] = tensor.to(dtype=torch.uint8)
            elif key.endswith(".weight_int8"):
                load_state[key] = tensor.to(dtype=torch.int8)
            elif key.endswith(".sigma_rows") or key.endswith(".mu_rows"):
                load_state[key] = tensor.to(dtype=torch.float16)
            elif tensor.is_floating_point():
                load_state[key] = tensor.to(dtype=torch.float32)
            else:
                load_state[key] = tensor

        model.load_state_dict(load_state, strict=True)
        model.eval()
        model.to(device)

        print("Materializing packed weights to FP16 for PyTorch cuBLAS compatibility...", flush=True)
        for module in model.modules():
            if hasattr(module, "materialize"):
                module.materialize()

        return model


# =============================================================================
# pollux_step — least-action crystallization optimizer
# =============================================================================


@torch.no_grad()
def pollux_step(
    state: PolluxState,
    model: nn.Module,
    *,
    ce_mean: float | torch.Tensor,
) -> None:
    """Execute one Pollux optimisation step via the thermodynamic estimator.

    No architectural hyperparameters; requires DATASET_NOISE_FLOOR (H_floor) as
    an empirically measured environmental boundary condition for the corpus.
    Updates continuous pre-weights via heat-modulated Adam dynamics scaled by
    the topological drag coefficient 1/C, applies width-dependent stability
    η_d = 1 + C ln(d / (24C)²), projects H24 tensors onto the Leech lattice
    once per step, and writes fully discrete structural centres to the
    observable backbone weights.  At the geometric reference baseline
    d* = 1152 = (24C)², η_d = 1.
    """
    param_dev = next(model.parameters()).device
    dev, dt = state.device, state.dtype
    z = lambda value: _z(dev, dt, value)
    d = float(state.n_embd.item())
    # Universal Pollux scaling (width-dependent representational stability). C is topological friction
    # against dimensional scaling: wider tensors concentrate mass in H24 space.
    # Width-inertia equilibrium d* = (24·C)² = 1152 for C = √2 (η_d = 1 at Pollux-1152 width).
    width_inertia = 1.0 + C * math.log(d / ((float(H24_DIM) * C) ** 2))
    H_MAX = entropy_ceiling(float(state.vocab_size.item()))

    # --- Macroscopic thermodynamics ---
    H_FLOOR = float(DATASET_NOISE_FLOOR)
    ce_mean_t = (
        ce_mean
        if isinstance(ce_mean, torch.Tensor)
        else torch.tensor(ce_mean, device=param_dev, dtype=torch.float32)
    )
    state.current_loss.copy_(ce_mean_t.detach().to(device=dev, dtype=dt))

    # Topological drag coefficient 1/C (= s_lattice): continuous updates are damped
    # by the Leech covering-radius barrier.  Stochastic batch noise is absorbed
    # by γ in the Adam β coefficients — no empirical cancellation term.
    #
    # Voronoi jitter floor (H_min): NSM of the H24 Voronoi cell (G24 = γ ≈ 0.065771).
    min_heat = float(gamma)
    instant_heat = ((ce_mean_t - H_FLOOR) / (H_MAX - H_FLOOR)).clamp(min_heat, 1.0)
    instant_on_state = instant_heat.detach().to(device=dev, dtype=dt)

    # Hardware-agnostic macroscopic time constant via equipartition: distribute
    # microscopic jitter γ evenly across the 24 Leech-lattice degrees of freedom
    # (α_EMA ≈ 0.00274 for γ = G₂₄).  Decoupled from grad_accum / batching.
    ema_alpha = float(gamma) / float(H24_DIM)
    if state.step.eq(0).all():
        state.macro_heat.copy_(instant_on_state)
    else:
        state.macro_heat.lerp_(instant_on_state, ema_alpha)

    heat_t = state.macro_heat.clone()

    # Adam momentum coefficients derived from γ: β₁ = 1 − √γ, β₂ = 1 − γ.
    beta_2 = 1.0 - float(gamma)
    beta_1 = 1.0 - math.sqrt(float(gamma))
    step_f = state.step.to(dtype=torch.float32, device=dev) + 1.0
    bc1 = 1.0 - (torch.as_tensor(beta_1, device=dev, dtype=torch.float32) ** step_f)
    bc2 = 1.0 - (torch.as_tensor(beta_2, device=dev, dtype=torch.float32) ** step_f)

    codebook_by_key: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    adam_energy_sum = torch.zeros((), device=dev, dtype=dt)
    param_count = torch.zeros((), device=dev, dtype=dt)

    for name, p in state.tracked_params.items():
        grad = p.grad
        if grad is None:
            continue

        semantic_grad = grad.detach().to(device=p.device, dtype=p.dtype)
        m, v = state.kinetic_buffers(name, p)
        latent = state.latent_buffer(name, p)
        m.mul_(beta_1).add_(semantic_grad, alpha=1.0 - beta_1)
        v.mul_(beta_2).addcmul_(semantic_grad, semantic_grad, value=1.0 - beta_2)

        bc1_on_p = bc1.to(device=p.device, dtype=p.dtype)
        bc2_on_p = bc2.to(device=p.device, dtype=p.dtype)
        m_unbiased = m / torch.clamp(bc1_on_p, min=NUMERICAL_EPSILON)
        v_unbiased = v / torch.clamp(bc2_on_p, min=NUMERICAL_EPSILON)

        heat_on_p = heat_t.to(device=p.device, dtype=p.dtype)
        denom = torch.sqrt(v_unbiased) + NUMERICAL_EPSILON

        # --- Isomorphic H24 update dynamics (width-dependent scaling) ---
        # Matrices (≥2D) accumulate variance across embedding width and inherit
        # width_inertia.  1D scalars (RMSNorm gains, Q/K norm gains) operate
        # elementwise and receive undamped pre-weight updates (stability factor = 1).
        is_matrix = p.dim() >= 2
        current_inertia = width_inertia if is_matrix else 1.0

        kinetic_scale = heat_on_p * float(gamma) / float(C)
        adam_step = -(kinetic_scale / current_inertia) * (m_unbiased / denom)

        if _is_h24_tensor(p.data):
            # Per-24D-atom kinematic speed cap: prevents topological teleportation
            # across discrete lattice gaps in a single update.
            adam_h24 = adam_step.reshape(-1, H24_DIM)
            step_norm = torch.linalg.vector_norm(adam_h24, dim=-1, keepdim=True)
            speed_cap = float(KINEMATIC_LIMIT) * float(s_lattice)
            adam_step = (
                adam_h24 * torch.clamp(speed_cap / (step_norm + NUMERICAL_EPSILON), max=1.0)
            ).reshape_as(adam_step)

            # Jitter-squared dissipation: γ² modulates latent cooling with heat.
            # The Snap offset supplies minimum gradient variance for gradients to
            # overcome discrete lattice gaps without thermodynamic freezing.
            weight_decay_factor = 1.0 - (heat_on_p * (float(gamma) ** 2))
            latent.mul_(weight_decay_factor)
            latent.add_(adam_step)

            # H24 projection (latent-parameter QAT): global row-wise RMS, lattice
            # snap, re-scale — executed once per parameter per optimiser step only.
            normed, sigma_rows = normalize_matrix_by_row_rms(latent)
            cb_key = (str(p.device), torch.float32)
            codebook = codebook_by_key.get(cb_key)
            if codebook is None:
                codebook = get_h24_codebook(p.device, torch.float32)
                codebook_by_key[cb_key] = codebook
            center_abs = castor_quantize_leech(normed / float(s_lattice), codebook)
            center = scale_matrix_by_row_rms(center_abs, sigma_rows) * float(s_lattice)
            p.data.copy_(center)
        else:
            # 1D semantic scalars: continuous pre-weights with geometric-slack clamp at unity.
            weight_decay_factor = 1.0 - (heat_on_p * (float(gamma) ** 2))
            latent.mul_(weight_decay_factor)
            latent.add_(adam_step)
            slack = float(GEOMETRIC_SLACK)
            latent.clamp_(1.0 - slack, 1.0 + slack)
            p.data.copy_(latent)

        # Aggregate global thermodynamic observables (device-resident, no .item() in loop).
        adam_energy_sum = adam_energy_sum + torch.linalg.vector_norm(
            adam_step.view(-1), ord=2
        ).to(dtype=dt)
        param_count = param_count + z(float(adam_step.numel()))

    safe_param_count = param_count.clamp(min=1.0)

    state.macro_heat.copy_(heat_t.to(device=dev, dtype=dt))
    state.adam_force.copy_(adam_energy_sum / safe_param_count)

    state.step.add_(1)
