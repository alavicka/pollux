#!/usr/bin/env python3
# Copyright (c) 2026 Alexander Lavicka.
# This source code is licensed under the PolyForm Noncommercial License 1.0.0.
# A copy of this license is available at https://polyformproject.org/licenses/noncommercial/1.0.0/
# Commercial utilization or hardware integration requires a separate license from the patent holder.
"""Compute held-out validation cross-entropy for Pollux training checkpoints.

Runs a forward-only pass over a reserved tail slice of the FineWeb-Edu memmap
(not used as a dedicated training split during memmap preparation). For each
Pollux-1152 and Pollux-1920 checkpoint at steps 5k, 10k, and 15k, evaluates
exactly NUM_VAL_BATCHES batches and reports mean cross-entropy loss.

Memory notes (RTX 5090 / WSL display GPU)
-----------------------------------------
Training checkpoints store model weights PLUS physics/Adam buffers (~4x params).
Those extra tensors are discarded on CPU and never moved to VRAM.

The LM head would otherwise materialise logits of shape [B, T, 50688]. That
tensor, then cast to float32 for CE, is the dominant activation. Validation
applies the head in short token chunks instead.

Usage
-----
    python validate.py
    python validate.py --batch-size 2 --seq-len 1024
    python validate.py --models pollux-1152 --data-bin data/fineweb_10b.bin
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Must be set before the first CUDA context is created.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pollux import (
    PolluxConfig,
    PolluxModel,
    config_from_checkpoint,
    h24_basis_fingerprint,
)
from train import (
    default_fineweb_bin_path,
    mask_inactive_vocab_logits,
    open_fineweb_memmap,
)

# =============================================================================
# Easily adjustable validation settings
# =============================================================================

# Match the training micro-batch (PolluxConfig.batch_size = 2). Larger values
# mainly inflate the LM-head logit tensor, not statistical quality.
BATCH_SIZE = 1
SEQ_LEN = 1024  # capped to each checkpoint's trained context length at runtime
NUM_VAL_BATCHES = 300
# Tokens per LM-head slice. 64 tokens × vocab 50688 × bf16 ≈ 6.5 MB / row.
LOGIT_CHUNK_TOKENS = 32
VAL_HOLDOUT_FRACTION = 0.05  # last 5 % of the memmap reserved for validation
VAL_SEED = 12345  # distinct from training seed (42) in train.py
# Leave compositor headroom on a Windows/WSL display GPU (TDR ~2 s).
CUDA_MEMORY_FRACTION = 0.90

CHECKPOINT_STEPS: tuple[int, ...] = (5000, 10000, 15000)

# Search order for checkpoint files (relative to publish/ unless absolute).
MODEL_VARIANTS: dict[str, dict[str, Any]] = {
    "pollux-1152": {
        "expected_n_embd": 1152,
        "search_dirs": (
            "checkpoints/pollux-1152",
            "checkpoints/pollux_1152",
            "checkpoints",
        ),
    },
    "pollux-1920": {
        "expected_n_embd": 1920,
        "search_dirs": (
            "checkpoints/pollux-1920",
            "checkpoints/pollux_1920",
            "checkpoints",
        ),
    },
}

DEFAULT_JSON_OUTPUT = _HERE / "val_loss_results.json"
DEFAULT_CSV_OUTPUT = _HERE / "val_loss_results.csv"


# =============================================================================
# CUDA / attention setup
# =============================================================================


def _vram_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_allocated(device)) / (1024.0 * 1024.0)


def _sdpa_context():
    """Prefer O(T) attention kernels; the math backend materialises [B, H, T, T]."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
        ]
        return sdpa_kernel(backends)
    except Exception:
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        return nullcontext()


def _probe_sdpa(device: torch.device) -> None:
    """Fail fast on a tiny tensor if no O(T) attention kernel exists (Blackwell)."""
    dummy = torch.zeros(1, 2, 8, 24, device=device, dtype=torch.bfloat16)
    try:
        with torch.inference_mode(), _sdpa_context():
            F.scaled_dot_product_attention(dummy, dummy, dummy, is_causal=True)
        return
    except Exception:
        pass
    torch.backends.cuda.enable_math_sdp(True)
    print(
        "Warning: flash/efficient SDPA unavailable; falling back to math attention. "
        "Keep --batch-size 2 and --logit-chunk-tokens 64.",
        flush=True,
    )


def configure_cuda(device: torch.device, *, memory_fraction: float) -> None:
    """Harden the CUDA context for validation on a display GPU."""
    if device.type != "cuda":
        return
    # Full FP32 matmul: TF32 would blur H24 lattice geometry (same as train.py).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    try:
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction), device)
    except Exception as exc:
        print(f"Warning: could not cap CUDA memory fraction: {exc}", flush=True)
    _probe_sdpa(device)
    print(
        f"CUDA memory cap: {100.0 * memory_fraction:.0f}% | "
        f"allocated {_vram_mb(device):.0f} MB",
        flush=True,
    )


# =============================================================================
# Held-out validation data loader
# =============================================================================


class ValidationHoldoutDataLoader:
    """Random batches sampled only from a tail holdout region of the memmap."""

    def __init__(
        self,
        bin_path: str,
        *,
        holdout_fraction: float = VAL_HOLDOUT_FRACTION,
        active_vocab: int | None = None,
        seed: int = VAL_SEED,
    ) -> None:
        self.bin_path = os.path.abspath(bin_path)
        self.data = open_fineweb_memmap(self.bin_path)
        self.token_count = int(self.data.shape[0])
        holdout_tokens = max(int(self.token_count * float(holdout_fraction)), 1)
        self.region_start = max(self.token_count - holdout_tokens, 0)
        self.region_end = self.token_count
        self.active_vocab = int(active_vocab) if active_vocab is not None else None
        self._rng = np.random.default_rng(seed)

    def get_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = torch.device(device)
        batch_size = int(batch_size)
        seq_len = int(seq_len)
        block_len = seq_len + 1
        min_start = int(self.region_start)
        max_start = int(self.region_end - block_len)
        if max_start < min_start:
            raise ValueError(
                f"Validation holdout in {self.bin_path} is too small: "
                f"region [{self.region_start}, {self.region_end}) needs at least "
                f"{block_len} tokens for seq_len={seq_len}."
            )

        starts = self._rng.integers(min_start, max_start + 1, size=batch_size)
        inp = np.empty((batch_size, seq_len), dtype=np.int64)
        tgt = np.empty((batch_size, seq_len), dtype=np.int64)
        active = self.active_vocab
        data = self.data
        for row, start in enumerate(starts):
            block = np.asarray(data[int(start) : int(start) + block_len], dtype=np.int64)
            if active is not None and active < 65536:
                np.clip(block, 0, active - 1, out=block)
            inp[row] = block[:-1]
            tgt[row] = block[1:]

        pin = dev.type == "cuda"
        x = torch.from_numpy(inp)
        y = torch.from_numpy(tgt)
        if pin:
            x = x.pin_memory()
            y = y.pin_memory()
        return x.to(dev, non_blocking=pin), y.to(dev, non_blocking=pin)


# =============================================================================
# Checkpoint discovery and loading
# =============================================================================


def _clean_state_dict(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        k = str(key)
        for prefix in ("_orig_mod.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        out[k] = value
    return out


def _checkpoint_filename_candidates(model_name: str, step: int) -> tuple[str, ...]:
    suffix = model_name.rsplit("-", 1)[-1]
    return (
        f"pollux_{suffix}_step_{step}.pt",
        f"pollux_{suffix}_step{step}.pt",
        f"pollux_step_{step}.pt",
        f"pollux_step{step}.pt",
    )


def resolve_checkpoint_path(
    model_name: str,
    step: int,
    *,
    search_dirs: Iterable[str],
) -> Path:
    """Locate a checkpoint file for ``model_name`` at training ``step``."""
    candidates: list[Path] = []
    for directory in search_dirs:
        root = Path(directory)
        if not root.is_absolute():
            root = _HERE / root
        for filename in _checkpoint_filename_candidates(model_name, step):
            candidates.append(root / filename)

    for path in candidates:
        if path.is_file():
            return path

    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"No checkpoint found for {model_name} at step {step}. Tried:\n{tried}"
    )


def _release_cpu() -> None:
    gc.collect()


def load_training_checkpoint(
    path: str | Path,
    device: torch.device,
    *,
    expected_n_embd: int | None = None,
) -> tuple[PolluxModel, PolluxConfig, int]:
    """Load only model weights from a .pt checkpoint; physics/Adam stay on disk.

    Training checkpoints are ~4× the live parameter footprint (weights +
    continuous pre-weights + Adam m/v). Loading that blob with
    ``map_location=cuda`` is enough to TDR a 32 GB display GPU.
    """
    ckpt_path = Path(path)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"{ckpt_path}: expected a dict with 'model_state_dict'.")

    ckpt_fp = ckpt.get("h24_basis_fingerprint")
    if ckpt_fp is not None and tuple(ckpt_fp) != h24_basis_fingerprint():
        raise ValueError(
            f"{ckpt_path}: H24 basis fingerprint mismatch with current runtime."
        )

    raw_state = ckpt["model_state_dict"]
    cfg = config_from_checkpoint(ckpt, state_dict=_clean_state_dict(raw_state))
    step = int(ckpt.get("step", 0))
    # Drop optimizer / kinetic tensors immediately — never copy them to GPU.
    ckpt.pop("physics_state_dict", None)
    del ckpt

    if expected_n_embd is not None and int(cfg.n_embd) != int(expected_n_embd):
        raise ValueError(
            f"{ckpt_path}: expected n_embd={expected_n_embd}, "
            f"checkpoint has n_embd={cfg.n_embd}."
        )

    load_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    state = {}
    for key, value in _clean_state_dict(raw_state).items():
        tensor = value.detach()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=load_dtype)
        state[key] = tensor
    del raw_state
    _release_cpu()

    model = PolluxModel(cfg)
    model.load_state_dict(state, strict=True)
    del state
    _release_cpu()

    model.to(device=device, dtype=load_dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(
        f"Loaded {ckpt_path.name} | n_embd={cfg.n_embd} | "
        f"VRAM {_vram_mb(device):.0f} MB",
        flush=True,
    )
    return model, cfg, step


# =============================================================================
# Validation loop
# =============================================================================


@dataclass(frozen=True)
class ValidationResult:
    model: str
    step: int
    val_loss: float
    checkpoint: str
    batches: int
    seq_len: int
    batch_size: int


def _backbone_hidden(model: PolluxModel, idx: torch.Tensor) -> torch.Tensor:
    """Decoder hidden states before the LM head — shape [B, T, n_embd]."""
    _, seq_len = idx.shape
    pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
    hidden = model.tok_emb(idx) + model.pos_emb(pos)
    for block in model.blocks:
        hidden = block(hidden)
    return model.ln_f(hidden)


def _chunked_cross_entropy(
    model: PolluxModel,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    *,
    active_vocab: int,
    chunk_tokens: int,
) -> torch.Tensor:
    """Mean CE without materialising the full [B, T, vocab] logit tensor."""
    _batch, seq_len, _width = hidden.shape
    chunk = max(int(chunk_tokens), 1)
    loss_sum = hidden.new_zeros((), dtype=torch.float32)
    token_count = 0
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        logits = model.head(hidden[:, start:end, :])
        logits = mask_inactive_vocab_logits(
            logits.reshape(-1, logits.size(-1)).float(),
            active_vocab,
        )
        chunk_targets = targets[:, start:end].reshape(-1)
        loss_sum = loss_sum + F.cross_entropy(logits, chunk_targets, reduction="sum")
        token_count += int(chunk_targets.numel())
        del logits, chunk_targets
    return loss_sum / max(token_count, 1)


def compute_validation_loss(
    model: PolluxModel,
    cfg: PolluxConfig,
    data_loader: ValidationHoldoutDataLoader,
    *,
    batch_size: int,
    seq_len: int,
    num_batches: int,
    logit_chunk_tokens: int,
    device: torch.device,
) -> float:
    """Mean cross-entropy over ``num_batches`` held-out validation batches."""
    effective_seq_len = min(int(seq_len), int(cfg.seq_len))
    active_vocab = int(cfg.tokenizer_vocab_size)

    loss_sum = 0.0
    with torch.inference_mode(), _sdpa_context():
        for _ in tqdm(range(int(num_batches)), desc="val batches", leave=False):
            inp, tgt = data_loader.get_batch(batch_size, effective_seq_len, device)
            hidden = _backbone_hidden(model, inp)
            loss = _chunked_cross_entropy(
                model,
                hidden,
                tgt,
                active_vocab=active_vocab,
                chunk_tokens=logit_chunk_tokens,
            )
            loss_sum += float(loss.item())
            del inp, tgt, hidden, loss

    return loss_sum / max(int(num_batches), 1)


def run_validation_suite(
    *,
    data_bin: str,
    batch_size: int,
    seq_len: int,
    num_batches: int,
    logit_chunk_tokens: int,
    device: torch.device,
    holdout_fraction: float,
    model_variants: dict[str, dict[str, Any]],
    checkpoint_steps: tuple[int, ...],
    memory_fraction: float,
) -> list[ValidationResult]:
    """Evaluate all requested checkpoints and return structured results."""
    configure_cuda(device, memory_fraction=memory_fraction)

    active_vocab = int(PolluxConfig.tokenizer_vocab_size)
    data_loader = ValidationHoldoutDataLoader(
        data_bin,
        holdout_fraction=holdout_fraction,
        active_vocab=active_vocab,
        seed=VAL_SEED,
    )
    print(
        f"Validation holdout: tokens [{data_loader.region_start:,}, "
        f"{data_loader.region_end:,}) "
        f"({100.0 * holdout_fraction:.1f}% tail of memmap)",
        flush=True,
    )

    results: list[ValidationResult] = []
    for model_name, spec in model_variants.items():
        expected_n_embd = int(spec["expected_n_embd"])
        search_dirs = spec.get("search_dirs", ("checkpoints",))

        for step in checkpoint_steps:
            ckpt_path = resolve_checkpoint_path(
                model_name,
                step,
                search_dirs=search_dirs,
            )
            print(f"\nLoading {model_name} checkpoint: {ckpt_path}", flush=True)
            model, cfg, ckpt_step = load_training_checkpoint(
                ckpt_path,
                device,
                expected_n_embd=expected_n_embd,
            )
            effective_step = ckpt_step or step
            effective_seq_len = min(int(seq_len), int(cfg.seq_len))

            val_loss = compute_validation_loss(
                model,
                cfg,
                data_loader,
                batch_size=batch_size,
                seq_len=effective_seq_len,
                num_batches=num_batches,
                logit_chunk_tokens=logit_chunk_tokens,
                device=device,
            )
            result = ValidationResult(
                model=model_name,
                step=effective_step,
                val_loss=val_loss,
                checkpoint=str(ckpt_path),
                batches=num_batches,
                seq_len=effective_seq_len,
                batch_size=batch_size,
            )
            results.append(result)
            print(
                f"{model_name} | Step {effective_step} | Val Loss: {val_loss:.4f} | "
                f"VRAM {_vram_mb(device):.0f} MB",
                flush=True,
            )

            del model
            _release_cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)

    return results


def save_results(
    results: list[ValidationResult],
    *,
    json_path: Path,
    csv_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Write validation metrics to JSON and CSV for downstream plotting."""
    payload = {
        "metadata": metadata,
        "results": [
            {
                "model": row.model,
                "step": row.step,
                "val_loss": row.val_loss,
                "checkpoint": row.checkpoint,
                "batches": row.batches,
                "seq_len": row.seq_len,
                "batch_size": row.batch_size,
            }
            for row in results
        ],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "step", "val_loss", "checkpoint", "batches", "seq_len", "batch_size"),
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "model": row.model,
                    "step": row.step,
                    "val_loss": f"{row.val_loss:.6f}",
                    "checkpoint": row.checkpoint,
                    "batches": row.batches,
                    "seq_len": row.seq_len,
                    "batch_size": row.batch_size,
                }
            )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute held-out validation loss for Pollux checkpoints.",
    )
    parser.add_argument(
        "--data-bin",
        default=default_fineweb_bin_path(),
        help="Path to FineWeb uint16 token memmap (default: data/fineweb_10b.bin)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--num-batches", type=int, default=NUM_VAL_BATCHES)
    parser.add_argument(
        "--logit-chunk-tokens",
        type=int,
        default=LOGIT_CHUNK_TOKENS,
        help="LM-head slice length; lower this first if VRAM still spikes",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=VAL_HOLDOUT_FRACTION,
        help="Fraction of memmap tokens reserved at the tail for validation",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda when available)",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=CUDA_MEMORY_FRACTION,
        help="CUDA per-process memory cap (default: 0.90, leaves display headroom)",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_JSON_OUTPUT),
        help="Path for JSON results",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_CSV_OUTPUT),
        help="Path for CSV results",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=list(CHECKPOINT_STEPS),
        help="Checkpoint steps to evaluate (default: 5000 10000 15000)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_VARIANTS.keys()),
        default=list(MODEL_VARIANTS.keys()),
        help="Which Pollux widths to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    selected_models = {name: MODEL_VARIANTS[name] for name in args.models}
    metadata = {
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "num_batches": int(args.num_batches),
        "logit_chunk_tokens": int(args.logit_chunk_tokens),
        "holdout_fraction": float(args.holdout_fraction),
        "val_seed": VAL_SEED,
        "data_bin": os.path.abspath(args.data_bin),
        "device": str(device),
        "steps": [int(step) for step in args.steps],
        "models": list(args.models),
        "memory_fraction": float(args.memory_fraction),
    }

    print("Pollux validation loss evaluation", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Data: {metadata['data_bin']}", flush=True)
    print(
        f"Batch size={metadata['batch_size']} | "
        f"seq_len={metadata['seq_len']} | "
        f"batches={metadata['num_batches']} | "
        f"logit_chunk={metadata['logit_chunk_tokens']}",
        flush=True,
    )

    results = run_validation_suite(
        data_bin=metadata["data_bin"],
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        num_batches=int(args.num_batches),
        logit_chunk_tokens=int(args.logit_chunk_tokens),
        device=device,
        holdout_fraction=float(args.holdout_fraction),
        model_variants=selected_models,
        checkpoint_steps=tuple(int(step) for step in args.steps),
        memory_fraction=float(args.memory_fraction),
    )

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    save_results(results, json_path=json_path, csv_path=csv_path, metadata=metadata)

    print("\nSummary", flush=True)
    for row in results:
        print(f"{row.model} | Step {row.step} | Val Loss: {row.val_loss:.4f}", flush=True)
    print(f"\nWrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
