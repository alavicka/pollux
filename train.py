#!/usr/bin/env python3
# Copyright (c) 2026 Alexander Lavicka.
# This source code is licensed under the PolyForm Noncommercial License 1.0.0.
# A copy of this license is available at https://polyformproject.org/licenses/noncommercial/1.0.0/
# Commercial utilization or hardware integration requires a separate license from the patent holder.
"""Pollux training entry point.

Executes least-action crystallization on the 24D Leech lattice with local 
FineWeb-Edu memmap ingestion, the endogenous thermodynamic estimator (pollux_step), 
and optional Weights & Biases telemetry.

The thermodynamic estimator (pollux_step) has no architectural hyperparameters
and requires no learning-rate schedule, weight decay, gradient clipping, or
warmup. It requires exactly one empirically measured environmental boundary condition:
H_floor (DATASET_NOISE_FLOOR in pollux.py) — the cross-entropy convergence
floor of the training corpus, analogous to ambient temperature in Carnot theory.

Usage:
    python train.py [--wandboff] [--resume PATH] [--auto-resume]
                    [--target-tokens N] [--data-bin PATH]

Dependencies:
    pip install torch numpy tiktoken wandb   # wandb is optional
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Checkpoint cadence in optimizer steps (independent of gradient accumulation).
_CHECKPOINT_INTERVAL = 2500

FINEWEB_BIN_NAME = "fineweb_10b.bin"
FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_SUBSET = "sample-10BT"

# GPT-2 tiktoken encoding name and its vocabulary size.
TIKTOKEN_ENCODING = "gpt2"
GPT2_VOCAB_SIZE = 50257

import numpy as np
import torch

# Fixed seed for memmap batch sampling and resume parity via skip_batches().
_TRAINING_SEED = 42


def set_seed(seed: int = _TRAINING_SEED) -> None:
    """Seed Python, NumPy, and PyTorch RNGs before dataloader/model initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Tokenizer (tiktoken GPT-2)
# =============================================================================


def load_tokenizer():
    """Return a tiktoken GPT-2 encoding object.

    The returned object exposes:
        .encode(text) -> list[int]
        .decode(ids)  -> str
        .eot_token    -> int   (end-of-text token id, equivalent to eos)
        .n_vocab      -> int   (50257)
    """
    import tiktoken
    enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return enc


def mask_inactive_vocab_logits(logits: torch.Tensor, active_vocab_size: int) -> torch.Tensor:
    """Mask padded vocabulary tail beyond the active tokenizer size to −∞."""
    active = int(active_vocab_size)
    if logits.size(-1) <= active:
        return logits
    return logits.masked_fill(
        torch.arange(logits.size(-1), device=logits.device) >= active,
        float("-inf"),
    )


# =============================================================================
# Dataset
# =============================================================================


def default_fineweb_bin_path() -> str:
    """Resolve path to the local FineWeb token binary (env override or repo default)."""
    env_path = os.environ.get("POLLUX_FINEWEB_BIN", "").strip()
    if env_path:
        return os.path.abspath(env_path)
    return os.path.join(_HERE, "data", FINEWEB_BIN_NAME)


def open_fineweb_memmap(path: str | None = None) -> np.memmap:
    """Open the FineWeb uint16 token stream as a read-only memory map."""
    bin_path = os.path.abspath(str(path or default_fineweb_bin_path()))
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(
            f"FineWeb token memmap not found: {bin_path}\n"
            "Build the corpus once with `python prepare_fineweb.py` (requires network access)."
        )
    return np.memmap(bin_path, dtype=np.uint16, mode="r")


class FineWebMemmapDataLoader:
    """O(1) random-access batches from a local uint16 token memmap."""

    def __init__(
        self,
        bin_path: str | None = None,
        *,
        active_vocab: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.bin_path = os.path.abspath(str(bin_path or default_fineweb_bin_path()))
        self.data = open_fineweb_memmap(self.bin_path)
        self.token_count = int(self.data.shape[0])
        self.active_vocab = int(active_vocab) if active_vocab is not None else None
        self._rng = np.random.default_rng(seed)

    def get_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``batch_size`` contiguous sequences of length ``seq_len`` (shifted targets)."""
        dev = torch.device(device)
        batch_size = int(batch_size)
        seq_len = int(seq_len)
        block_len = seq_len + 1
        max_start = self.token_count - block_len
        if max_start < 0:
            raise ValueError(
                f"Token memmap {self.bin_path} contains {self.token_count} tokens; "
                f"at least {block_len} are required for seq_len={seq_len}."
            )

        starts = self._rng.integers(0, max_start + 1, size=batch_size)
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

    def skip_batches(self, count: int, *, batch_size: int, seq_len: int, device: torch.device | str) -> None:
        """Advance the RNG stream by ``count`` batches (resume parity after checkpoint)."""
        for _ in range(int(count)):
            self.get_batch(batch_size, seq_len, device)


# =============================================================================
# Training loop
# =============================================================================


def _clear_grads(model: object) -> None:
    """Release gradient tensors without zero-fill (reduces allocator pressure on long runs)."""
    for p in getattr(model, "parameters")():
        p.grad = None


def _train_loop(
    *,
    use_wandb: bool,
    resume_path: str | None,
    target_tokens: int | None,
    data_bin: str | None,
) -> None:
    """Main training loop: forward, backward, pollux_step, logging, checkpointing."""
    import torch.nn.functional as F
    from torch import amp as torch_amp

    set_seed(_TRAINING_SEED)

    from pollux import (
        G24,
        H24_DIM,
        C,
        CRITICAL_DIM,
        gamma,
        KINEMATIC_LIMIT,
        s_lattice,
        PolluxConfig,
        PolluxModel,
        PolluxState,
        pollux_step,
    )
    from castor import H24_CODEBOOK_COUNT, get_h24_codebook

    # Full FP32 matmul on CUDA: TF32 would blur H24 lattice geometry.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    PolluxConfig.validate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(
        "Pollux | least-action crystallization | FineWeb-Edu 10BT memmap | tiktoken GPT-2",
        flush=True,
    )

    train_cfg = PolluxConfig()
    bin_path = os.path.abspath(str(data_bin or default_fineweb_bin_path()))

    active_vocab = int(PolluxConfig.tokenizer_vocab_size)

    print(f"Token memmap: {bin_path}", flush=True)
    data_loader = FineWebMemmapDataLoader(bin_path, active_vocab=active_vocab, seed=_TRAINING_SEED)
    batch_size = int(PolluxConfig.batch_size)
    seq_len = int(PolluxConfig.seq_len)

    model = PolluxModel(train_cfg).to(device)
    if getattr(PolluxConfig, "use_compile", False) and device.type == "cuda":
        model = torch.compile(model, mode="default")
    model.train()

    state = PolluxState(device=device, dtype=torch.float32)
    state.bind_model_buffers(model)
    state.grad_accum_steps.copy_(
        torch.as_tensor(int(PolluxConfig.grad_accum_steps), device=device, dtype=torch.long)
    )

    wandb = None
    wandb_run = None
    if use_wandb:
        try:
            import wandb as wandb_mod

            wandb = wandb_mod
            wandb_run = wandb.init(
                project="pollux",
                config={
                    "friction_c": float(C),
                    "G24": float(G24),
                    "gamma": float(gamma),
                    "KINEMATIC_LIMIT": float(KINEMATIC_LIMIT),
                    "s_lattice": float(s_lattice),
                    "critical_dim": float(CRITICAL_DIM),
                    "H_min": float(gamma),
                    "H24_DIM": int(H24_DIM),
                    "n_layers": int(PolluxConfig.n_layers),
                    "n_heads": int(PolluxConfig.n_heads),
                    "n_embd": int(PolluxConfig.n_embd),
                    "vocab_size": int(PolluxConfig.vocab_size),
                    "tokenizer_vocab_size": int(PolluxConfig.tokenizer_vocab_size),
                    "batch_size": int(PolluxConfig.batch_size),
                    "seq_len": int(PolluxConfig.seq_len),
                    "grad_accum_steps": int(PolluxConfig.grad_accum_steps),
                    "dataset": FINEWEB_BIN_NAME,
                    "dataset_path": bin_path,
                    "dataset_source": f"{FINEWEB_DATASET}/{FINEWEB_SUBSET}",
                    "tokenizer": TIKTOKEN_ENCODING,
                },
            )
        except Exception as exc:
            print(
                f"W&B initialization failed ({exc}); continuing without experiment logging.",
                flush=True,
            )
            wandb = None
            wandb_run = None

    checkpoints_dir = os.path.join(_HERE, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    tokens_seen = 0
    resume_path = str(resume_path or "").strip()
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}", flush=True)
        try:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(resume_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise ValueError("Checkpoint file must contain a serialized mapping.")
        model.load_state_dict(ckpt["model_state_dict"])
        physics = ckpt.get("physics_state_dict")
        if physics is None:
            raise ValueError(
                "Checkpoint is missing physics_state_dict (Adam-momentum and thermodynamic state)."
            )
        state.load_state_dict(physics)
        state.grad_accum_steps.copy_(
            torch.as_tensor(int(PolluxConfig.grad_accum_steps), device=device, dtype=torch.long)
        )
        saved_step = int(ckpt.get("step", int(state.step.item())))
        state.step.copy_(torch.as_tensor(saved_step, device=device, dtype=torch.long))
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        batches_to_skip = saved_step * int(PolluxConfig.grad_accum_steps)
        if batches_to_skip > 0:
            print(f"Advancing memmap sampler by {batches_to_skip} batches...", flush=True)
            data_loader.skip_batches(
                batches_to_skip,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
            )
            print("Memmap sampler aligned with optimizer step.", flush=True)

        # --- FREE VRAM SPIKE ---
        del ckpt
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # -----------------------

    codebook = get_h24_codebook(device, torch.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    codebook_mb = codebook.element_size() * codebook.numel() / (1024 * 1024)
    print(
        f"Leech codebook ready on {device}: {H24_CODEBOOK_COUNT} entries ({codebook_mb:.1f} MB in FP32 for training)",
        flush=True,
    )

    target = int(target_tokens if target_tokens is not None else PolluxConfig.target_tokens)
    try:
        while tokens_seen < target:
            _clear_grads(model)
            ce_running = torch.zeros((), device=device, dtype=torch.float64)
            for _ in range(int(PolluxConfig.grad_accum_steps)):
                inp, tgt = data_loader.get_batch(batch_size, seq_len, device)
                if use_amp:
                    with torch_amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(inp)
                        logits = mask_inactive_vocab_logits(
                            logits.view(-1, logits.size(-1)).float(),
                            active_vocab,
                        )
                        ce = F.cross_entropy(logits, tgt.view(-1))
                else:
                    logits = model(inp)
                    logits = mask_inactive_vocab_logits(
                        logits.view(-1, logits.size(-1)).float(),
                        active_vocab,
                    )
                    ce = F.cross_entropy(logits, tgt.view(-1))
                (ce / float(PolluxConfig.grad_accum_steps)).backward()
                ce_running = ce_running + ce.detach().double()
                tokens_seen += int(inp.numel())
                del logits, ce, tgt, inp

            ce_mean = float((ce_running / max(int(PolluxConfig.grad_accum_steps), 1)).item())
            pollux_step(state, model, ce_mean=ce_mean)
            _clear_grads(model)

            step = int(state.step.item())
            print(
                f"Pollux | step {step} | loss {ce_mean:.4f} "
                f"| adam_force {float(state.adam_force.item()):.4e}",
                flush=True,
            )

            if wandb is not None and wandb_run is not None:
                wandb.log(state.metrics_for_wandb(), step=step)

            if step > 0 and step % _CHECKPOINT_INTERVAL == 0:
                ckpt_path = os.path.join(checkpoints_dir, f"pollux_step_{step}.pt")
                torch.save(
                    {
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "model_state_dict": getattr(model, "_orig_mod", model).state_dict(),
                        "physics_state_dict": state.state_dict(),
                        "config": train_cfg.to_dict(),
                    },
                    ckpt_path,
                )
                print(f"Checkpoint written: {ckpt_path}", flush=True)

        final_path = os.path.join(checkpoints_dir, "pollux_final.pt")
        torch.save(
            {
                "step": int(state.step.item()),
                "tokens_seen": tokens_seen,
                "model_state_dict": getattr(model, "_orig_mod", model).state_dict(),
                "physics_state_dict": state.state_dict(),
                "config": train_cfg.to_dict(),
            },
            final_path,
        )
        print(f"Final checkpoint written: {final_path}", flush=True)
    finally:
        if wandb is not None and wandb_run is not None:
            wandb.finish()


def _latest_checkpoint(checkpoints_dir: str) -> str:
    """Return the path of the most recent step checkpoint in ``checkpoints_dir``, or empty string."""
    paths = sorted(glob.glob(os.path.join(checkpoints_dir, "pollux_step_*.pt")))
    return paths[-1] if paths else ""


def main() -> None:
    """Parse CLI arguments and launch the Pollux training loop."""
    parser = argparse.ArgumentParser(
        description="Train the Pollux Architecture via least-action H24 crystallization"
    )
    parser.add_argument(
        "--wandboff",
        action="store_true",
        help="Disable Weights & Biases logging (enabled by default)",
    )
    parser.add_argument("--resume", default="", help="Path to a checkpoint to resume from")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from the latest step checkpoint in ./checkpoints/",
    )
    parser.add_argument("--target-tokens", type=int, default=None, help="Training token budget")
    parser.add_argument(
        "--data-bin",
        default="",
        help="Path to local fineweb_10b.bin memmap (default: ./data/fineweb_10b.bin)",
    )
    args = parser.parse_args()

    resume_path = str(args.resume or "").strip()
    if not resume_path and args.auto_resume:
        resume_path = _latest_checkpoint(os.path.join(_HERE, "checkpoints"))

    try:
        _train_loop(
            use_wandb=not bool(args.wandboff),
            resume_path=resume_path or None,
            target_tokens=args.target_tokens,
            data_bin=str(args.data_bin or "").strip() or None,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    os._exit(0)


if __name__ == "__main__":
    main()
