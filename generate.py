#!/usr/bin/env python3
# Copyright (c) 2026 Alexander Lavicka.
# This source code is licensed under the PolyForm Noncommercial License 1.0.0.
# A copy of this license is available at https://polyformproject.org/licenses/noncommercial/1.0.0/
# Commercial utilization or hardware integration requires a separate license from the patent holder.
"""Pollux inference engine — loads .plx or .pt checkpoints and generates text.

Packed .plx weights (≈27MB backbone SRAM for Pollux-1152) can reside
entirely in on-chip memory alongside the ≈9 MB FP16 Leech codebook. A
native forward pass is matrix-free: dense O(N²) weight-matrix matmul is
replaced by SRAM index lookup, FP16 σ_rms scale application, and
scalar–vector multiply–accumulate against continuous activations.

Runtime note: this reference script materialises packed 18-bit indices to
dense FP16 weight matrices via index_select, then calls F.linear (cuBLAS).
This validates crystallisation and zero-shot benchmarks but does NOT deliver
native SRAM-bound latency. True memory-bandwidth-bound inference requires
native matrix-free LUT gather–accumulate kernels.

Qualitative generation note: When generating unconditional text, Pollux
will produce grammatically flawless but factually absurd concepts. This
structural hallucination is not a bug, but the intended mechanical outcome
of the H24 Voronoi filter. By rejecting high-entropy factual trivia during
training, Pollux acts as a stateless cognitive CPU. This restriction is
designed for zero-interference Retrieval-Augmented Generation (RAG), forcing
the model to ground its fluid reasoning entirely in the provided external
prompt context rather than parametric memory.

Usage
-----
    python generate.py                           # interactive wizard
    python generate.py model.plx
    python generate.py model.plx --prompt "The second law of thermodynamics states"
    python generate.py model.pt --temperature 1.0 --top-k 50 --max-new-tokens 200
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pollux import (
    PolluxConfig,
    PolluxModel,
    codebook_fingerprint,
    config_from_checkpoint,
    h24_basis_fingerprint,
)

# =============================================================================
# .plx reader (private copy — Option A: script autonomy)
# =============================================================================

_PLX_MAGIC: bytes = b"POLLUXH2"
_PLX_VERSION: int = 0x04

_ID_TO_TORCH_DTYPE: dict[int, torch.dtype] = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.float16,
    3: torch.float32,
}


def _read_plx(path: str) -> dict[str, Any]:
    """Deserialise a .plx file into a payload dict accepted by
    PolluxModel.from_packed_checkpoint().

    Layout (mirrors pack.py _write_plx)
    ------------------------------------
    HEADER(16) CONFIG_LEN(4) CONFIG_JSON PAD FINGERPRINT(64)
    TENSOR_COUNT(4) [TENSOR]+
    """
    import numpy as np

    _ID_TO_NUMPY = {
        0: np.dtype("uint8"),
        1: np.dtype("int8"),
        2: np.dtype("float16"),
        3: np.dtype("float32"),
    }

    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != _PLX_MAGIC:
            raise ValueError(
                f"{path!r} is not a .plx file (bad magic {magic!r})."
            )
        (version,) = struct.unpack("B7x", f.read(8))
        if version != _PLX_VERSION:
            raise ValueError(
                f"Unsupported .plx version {version} (expected {_PLX_VERSION})."
            )

        # CONFIG
        (config_len,) = struct.unpack("<I", f.read(4))
        config_json = f.read(config_len).decode("utf-8")
        consumed = 16 + 4 + config_len
        pad = (64 - consumed % 64) % 64
        f.read(pad)

        # FINGERPRINT
        fp_values: tuple[float, ...] = struct.unpack("<16f", f.read(64))

        # TENSOR_COUNT
        (n_tensors,) = struct.unpack("<I", f.read(4))

        state_dict: dict[str, torch.Tensor] = {}
        for _ in range(n_tensors):
            (name_len,) = struct.unpack("<H", f.read(2))
            name = f.read(name_len).decode("utf-8")
            dtype_id, ndim = struct.unpack("BB", f.read(2))
            shape = list(struct.unpack(f"<{ndim}I", f.read(ndim * 4)))
            (data_len,) = struct.unpack("<Q", f.read(8))
            raw = f.read(data_len)
            np_dtype = _ID_TO_NUMPY.get(dtype_id, np.dtype("float32"))
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
            state_dict[name] = torch.from_numpy(arr)

    return {
        "format": "pollux_packed_v4",
        "config": json.loads(config_json),
        "codebook_fingerprint": fp_values,
        "state_dict": state_dict,
    }


# =============================================================================
# Model loading
# =============================================================================

def _clean_state_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Strip torch.compile / DataParallel prefixes from state dict keys."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        k = str(key)
        for prefix in ("_orig_mod.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = value
    return out


def load_model(
    path: str,
    device: torch.device,
) -> tuple[PolluxModel, PolluxConfig]:
    """Load a Pollux model from a .plx packed file or a .pt training checkpoint.

    For .plx files
        PolluxModel.from_packed_checkpoint() is called, which validates the
        codebook fingerprint and calls materialize() on every PackedH24Linear —
        expanding uint8 index arrays into FP16 weight matrices via a single
        codebook.index_select call, then F.linear on the materialised weights.

    For .pt files
        The full-precision training checkpoint is loaded, fingerprint-validated,
        and standard load_state_dict is called.
    """
    if path.endswith(".plx"):
        payload = _read_plx(path)
        model = PolluxModel.from_packed_checkpoint(path, device, payload=payload)
        cfg = PolluxConfig.from_dict(payload["config"])
        return model, cfg

    # .pt training checkpoint path
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"{path!r}: expected a dict with 'model_state_dict'; "
            "provide a Pollux training checkpoint."
        )

    ckpt_fp = ckpt.get("h24_basis_fingerprint")
    if ckpt_fp is not None and tuple(ckpt_fp) != h24_basis_fingerprint():
        raise ValueError(
            "H24 basis fingerprint mismatch: checkpoint was trained with a "
            "different lattice generator than the current runtime."
        )

    state = _clean_state_dict(ckpt["model_state_dict"])
    cfg = config_from_checkpoint(ckpt, state_dict=state)
    model = PolluxModel(cfg)

    load_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    state = {
        k: v.to(device=device, dtype=load_dtype)
        if v.is_floating_point() else v.to(device=device)
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, cfg


# =============================================================================
# Tokenizer (tiktoken GPT-2)
# =============================================================================

def _load_tokenizer():
    """Return a tiktoken GPT-2 encoding.  Exposes .encode(), .decode(), .eot_token."""
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            "tiktoken is required for generate.py.  "
            "Install it with: pip install tiktoken"
        ) from exc
    return tiktoken.get_encoding("gpt2")


# =============================================================================
# Vocabulary masking (inlined — no train.py import)
# =============================================================================

def _mask_inactive_vocab(logits: torch.Tensor, active_vocab: int) -> torch.Tensor:
    """Set logits beyond the active tokenizer vocabulary to −∞."""
    if logits.size(-1) <= active_vocab:
        return logits
    mask = torch.arange(logits.size(-1), device=logits.device) >= active_vocab
    return logits.masked_fill(mask, float("-inf"))


# =============================================================================
# Sampling helpers
# =============================================================================

def _apply_repetition_penalty(
    logits: torch.Tensor,
    ids: torch.Tensor,
    penalty: float,
) -> None:
    """In-place repetition penalty: divide positive logits, multiply negative."""
    if penalty <= 1.0:
        return
    seen = torch.unique(ids)
    idx = seen[(seen >= 0) & (seen < logits.numel())]
    if idx.numel() == 0:
        return
    vals = logits[idx]
    logits[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    _, keep = torch.topk(logits, k)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(0, keep, False)
    return logits.masked_fill(mask, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0 or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(0, sorted_idx, remove)
    return logits.masked_fill(mask, float("-inf"))


# =============================================================================
# Generation
# =============================================================================

@torch.inference_mode()
def generate(
    model: PolluxModel,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 80,
    temperature: float = 0.6,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    active_vocab: int | None = None,
) -> str:
    """Autoregressively generate text from a prompt.

    The active_vocab cap masks the padded vocabulary tail produced by Pollux's
    H24-aligned vocab_size (50,688) to the actual tokenizer vocabulary (50,257).
    """
    device = next(model.parameters()).device
    eos_id: int = int(getattr(tokenizer, "eot_token", 50256))
    v_active = active_vocab or eos_id + 1

    ids: list[int] = tokenizer.encode(prompt)
    ids = [t for t in ids if 0 <= t < v_active] or [eos_id]

    idx = torch.tensor([ids], device=device, dtype=torch.long)

    for _ in range(max_new_tokens):
        seq_len = int(getattr(model.cfg, "seq_len", 1024))
        logits = model(idx[:, -seq_len:])[:, -1, :].float()[0]   # [V]
        logits = _mask_inactive_vocab(logits, v_active)
        _apply_repetition_penalty(logits, idx[0], repetition_penalty)
        if temperature > 0:
            logits = logits / max(float(temperature), 1e-8)
        logits = _apply_top_k(logits, top_k)
        logits = _apply_top_p(logits, top_p)
        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos_id:
            break

    return tokenizer.decode(idx[0].tolist()).strip()


# =============================================================================
# Checkpoint discovery
# =============================================================================

_DEFAULT_PROMPTS: list[str] = [
    "The mitochondrion is a double-membrane-bound organelle. Its primary function is",
    "According to the theory of general relativity, gravity is not a traditional force, "
    "but rather a consequence of",
    "The fundamental theorem of calculus establishes a connection between differentiation "
    "and integration. Specifically, it states that",
    "Plate tectonics explains the movement of the Earth's lithosphere. "
    "When two tectonic plates collide at a convergent boundary,",
    "Vaccines stimulate the human immune system to recognise and fight pathogens. "
    "They typically trigger",
]


def _find_checkpoints(ckpt_dir: str) -> list[str]:
    from glob import glob
    patterns = ("*.plx", "*.pt")
    paths: list[str] = []
    for pat in patterns:
        paths.extend(
            p for p in glob(os.path.join(ckpt_dir, pat))
            if not p.endswith(".packed.pt")
        )

    def _sort_key(p: str) -> tuple[int, str]:
        base = os.path.basename(p)
        digits = "".join(c if c.isdigit() else " " for c in base).split()
        return (int(digits[-1]) if digits else -1, base)

    return sorted(dict.fromkeys(paths), key=_sort_key)


def _resolve_path(explicit: str | None, ckpt_dir: str) -> str:
    if explicit:
        if os.path.isfile(explicit):
            return os.path.abspath(explicit)
        for ckpt_dir_candidate in (ckpt_dir, str(_HERE)):
            candidate = os.path.join(ckpt_dir_candidate, explicit)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"Checkpoint not found: {explicit!r}")

    paths = _find_checkpoints(ckpt_dir)
    if not paths:
        raise FileNotFoundError(
            f"No .plx or .pt checkpoints found in {ckpt_dir!r}.  "
            "Run pack.py to create a .plx file, or train.py to create a .pt checkpoint."
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
        description="Pollux inference engine (loads .plx or .pt, generates text).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate.py\n"
            "  python generate.py model.plx\n"
            '  python generate.py model.plx --prompt "Quantum entanglement is"\n'
            "  python generate.py model.pt --temperature 1.0 --top-k 50\n"
        ),
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="",
        help="Path to .plx or .pt checkpoint (interactive wizard if omitted)",
    )
    parser.add_argument(
        "--prompt", "-p",
        action="append",
        default=[],
        metavar="TEXT",
        help="Prompt to complete (repeatable; default: built-in educational set)",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--device",
        default="",
        help='PyTorch device (default: "cuda" if available, else "cpu")',
    )
    parser.add_argument(
        "--ckpt-dir",
        default="",
        help="Directory to search for checkpoints (default: ./checkpoints)",
    )
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir or os.path.join(str(_HERE), "checkpoints")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    src = _resolve_path(str(args.checkpoint).strip() or None, ckpt_dir)
    print(f"\nLoading  {src}  on {device.type} ...", flush=True)

    model, cfg = load_model(src, device)
    tokenizer = _load_tokenizer()
    active_vocab = int(getattr(cfg, "tokenizer_vocab_size", 50257))

    step = 0
    if src.endswith(".pt"):
        try:
            step = int(
                torch.load(src, map_location="cpu", weights_only=False).get("step", 0)
            )
        except Exception:
            pass

    fmt = "packed .plx" if src.endswith(".plx") else f".pt (step {step:,})"
    print(
        f"Model:  {fmt} | layers={cfg.n_layers}  embd={cfg.n_embd}  "
        f"heads={cfg.n_heads}\n"
        f"Device: {device.type}  |  active vocab: {active_vocab:,}\n",
        flush=True,
    )

    prompts: list[str] = args.prompt or _DEFAULT_PROMPTS
    sep = "-" * 60

    for prompt in prompts:
        print(f"\nPrompt: {prompt}", flush=True)
        output = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            active_vocab=active_vocab,
        )
        print(f"Pollux: {output}", flush=True)
        print(sep, flush=True)


if __name__ == "__main__":
    main()
