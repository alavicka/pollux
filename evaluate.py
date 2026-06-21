#!/usr/bin/env python3
# Copyright (c) 2026 Alexander Lavicka.
# This source code is licensed under the PolyForm Noncommercial License 1.0.0.
# A copy of this license is available at https://polyformproject.org/licenses/noncommercial/1.0.0/
# Commercial utilization or hardware integration requires a separate license from the patent holder.
"""Evaluate a Pollux checkpoint with EleutherAI's lm-evaluation-harness.

Accepts both continuous .pt training checkpoints and 0.76-bit .plx packed
files. Results are printed in a stratified two-section table designed to
validate the H24 Voronoi confinement bottleneck (Section 4 of the paper):

  STRUCTURAL — fluid intelligence (BLiMP tasks)
  FACTUAL    — crystallised intelligence (SciQ, HellaSwag, PIQA, WinoGrande)

Note: All evaluation results reported in the paper were generated using
the `--fullblimp` option to run the complete 67-task BLiMP suite alongside
the factual benchmarks. The default run uses a 4-task structural subset for
quicker verification.

Runtime note: the current PyTorch evaluation path materialises packed 18-bit
indices to dense FP16 weight matrices via index_select, then runs standard
F.linear / cuBLAS. This validates cognitive capacity and factual filtering
but does NOT deliver native SRAM-bound latency. True memory-bandwidth-bound
execution requires native matrix-free LUT gather–accumulate kernels.

Usage
-----
    python evaluate.py                          # interactive wizard
    python evaluate.py model.plx
    python evaluate.py model.pt --batch-size 32
    python evaluate.py model.plx --tasks blimp_wh_island piqa
    python evaluate.py model.plx --limit 0.1   # 10 % of each task for testing
    python evaluate.py model.plx --fullblimp   # 67 BLiMP + 4 factual
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from lm_eval.api.model import TemplateLM as _TemplateLMBase
except ImportError:
    _TemplateLMBase = object  # type: ignore[assignment,misc]

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
# Vocabulary masking (inlined — no train.py import)
# =============================================================================

def _mask_inactive_vocab(logits: torch.Tensor, active_vocab: int) -> torch.Tensor:
    """Set logits beyond the active tokenizer vocabulary to −∞."""
    if logits.size(-1) <= active_vocab:
        return logits
    mask = torch.arange(logits.size(-1), device=logits.device) >= active_vocab
    return logits.masked_fill(mask, float("-inf"))


# =============================================================================
# lm-eval compatibility shims
# =============================================================================

def _import_pad_and_concat():
    """Import pad_and_concat, falling back to a self-contained implementation."""
    try:
        from lm_eval.models.utils_hf import pad_and_concat as _pac
        return _pac
    except ImportError:
        pass

    def pad_and_concat(
        max_length: int,
        tensors: list[torch.Tensor],
        padding_side: str = "right",
    ) -> torch.Tensor:
        out = []
        for t in tensors:
            pad = max_length - t.shape[0]
            if pad > 0:
                p = torch.zeros(pad, dtype=t.dtype, device=t.device)
                t = torch.cat([t, p], dim=0) if padding_side == "right" else torch.cat([p, t], dim=0)
            out.append(t)
        return torch.stack(out, dim=0)

    return pad_and_concat


# =============================================================================
# Model loading
# =============================================================================

def _clean_state_dict(raw: dict[str, Any]) -> dict[str, Any]:
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
    """Load a Pollux model from a .plx packed file or a .pt training checkpoint."""
    if path.endswith(".plx"):
        payload = _read_plx(path)
        model = PolluxModel.from_packed_checkpoint(path, device, payload=payload)
        cfg = PolluxConfig.from_dict(payload["config"])
        return model, cfg

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"{path!r}: expected a dict with 'model_state_dict'."
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
# Task catalogue
# =============================================================================

# Structural tasks (4): pure BLiMP minimal-pair acceptability judgements.
# Probe invariant syntactic rules encoded permanently in the H24 lattice.
# Expected: high accuracy (fluid intelligence — lattice-encoded logic).
_STRUCTURAL_TASKS: tuple[str, ...] = (
    "blimp_anaphor_gender_agreement",
    "blimp_regular_plural_subject_verb_agreement_1",
    "blimp_wh_island",
    "blimp_superlative_quantifiers_1",
)

# Factual tasks (4): pure multiple-choice world-knowledge benchmarks.
# Probe crystallised associations whose high-entropy gradients collapse to
# the null attractor — expected near random chance in a Pollux model.
_FACTUAL_TASKS: tuple[str, ...] = (
    "hellaswag",
    "winogrande",
    "sciq",
    "piqa",
)

DEFAULT_TASKS: tuple[str, ...] = _STRUCTURAL_TASKS + _FACTUAL_TASKS

TASK_ALIASES: dict[str, str] = {
    "blimp_anaphor_agreement":    "blimp_anaphor_gender_agreement",
    "blimp_subject_verb":         "blimp_regular_plural_subject_verb_agreement_1",
    "blimp_island_effects":       "blimp_wh_island",
    "blimp_quantifiers":          "blimp_superlative_quantifiers_1",
}


def resolve_tasks(names: Sequence[str]) -> list[str]:
    return [TASK_ALIASES.get(str(n).strip(), str(n).strip()) for n in names]


def build_eval_tasks(
    *,
    explicit_tasks: Sequence[str] | None,
    fullblimp: bool,
) -> tuple[list[str], list[str] | None]:
    """Return (tasks for lm-eval, structural task keys for table or None).

    When *structural* is ``None``, structural rows are inferred from results
    after evaluation (required for the ``blimp`` task group).
    """
    if explicit_tasks is not None:
        tasks = resolve_tasks(explicit_tasks)
        structural = [
            t for t in tasks
            if t == "blimp" or t.startswith("blimp_")
        ]
        return tasks, structural or None

    if fullblimp:
        tasks = ["blimp", *list(_FACTUAL_TASKS)]
        return tasks, None

    tasks = [
        *list(_STRUCTURAL_TASKS),
        *list(_FACTUAL_TASKS),
    ]
    return tasks, list(_STRUCTURAL_TASKS)


def structural_tasks_from_results(
    task_results: dict[str, Any],
    structural_hint: Sequence[str] | None,
) -> list[str]:
    """Resolve structural task keys present in an lm-eval results dict."""
    if structural_hint is not None:
        return [t for t in structural_hint if t in task_results]
    return sorted(
        t for t in task_results
        if t.startswith("blimp_") or t == "blimp"
    )


# =============================================================================
# lm-eval adapter
# =============================================================================

class PolluxLM(_TemplateLMBase):
    """lm-evaluation-harness adapter for a Pollux causal language model."""

    backend = "causal"

    def __init__(
        self,
        model: PolluxModel,
        tokenizer,
        cfg: PolluxConfig,
        *,
        device: torch.device,
        batch_size: int | str = 16,
        max_batch_size: int = 64,
    ) -> None:
        super().__init__()

        self._model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self._device = device
        self.max_batch_size = int(max_batch_size)
        self.batch_sizes: dict[int, int] = {}
        self.batch_schedule = 1.0

        if str(batch_size).startswith("auto"):
            parts = str(batch_size).split(":")
            self.batch_size: int | str = parts[0]
            if len(parts) > 1:
                self.batch_schedule = float(parts[1])
        else:
            self.batch_size = int(batch_size)

        self.max_length = int(cfg.seq_len)
        self.active_vocab = int(cfg.tokenizer_vocab_size)
        self._rank = 0
        self._world_size = 1
        self._model.eval()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def eot_token_id(self) -> int:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            raise ValueError("Tokenizer must define eos_token_id.")
        return int(eos)

    @property
    def tokenizer_name(self) -> str:
        return "gpt2"

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        **kwargs: Any,
    ) -> list[int]:
        if add_special_tokens is False:
            return self.tokenizer.encode(
                string, add_special_tokens=False
            )
        return self.tokenizer.encode(string, add_special_tokens=True)

    def tok_decode(
        self, tokens: list[int], skip_special_tokens: bool = True
    ) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    @torch.inference_mode()
    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        logits = self._model(inps)
        return _mask_inactive_vocab(logits, self.active_vocab)

    def _detect_batch_size(
        self,
        requests: Sequence[tuple[Any, list[int], list[int]]] | None = None,
        pos: int = 0,
    ) -> int:
        """Binary-search the largest executable batch size without OOM."""
        if requests:
            _, ctx, cont = requests[pos]
            probe_len = len((ctx + cont)[-(self.max_length + 1):][:-1])
        else:
            probe_len = self.max_length

        def _probe(bs: int) -> bool:
            dummy = torch.ones((bs, probe_len), device=self.device, dtype=torch.long)
            for _ in range(2):
                F.log_softmax(self._model_call(dummy).float(), dim=-1)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return True

        best = 1
        probe = 1
        while probe <= self.max_batch_size:
            try:
                _probe(probe)
                best = probe
                if probe == self.max_batch_size:
                    break
                probe = min(probe * 2, self.max_batch_size)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or "cuda error" in msg:
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                raise

        if best < probe and probe > best + 1:
            lo, hi = best + 1, min(probe - 1, self.max_batch_size)
            while lo <= hi:
                mid = (lo + hi) // 2
                try:
                    _probe(mid)
                    best = mid
                    lo = mid + 1
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "out of memory" in msg or "cuda error" in msg:
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        hi = mid - 1
                    else:
                        raise
        return best

    def _batch_scheduler(self, pos: int, n_reordered: Iterable[Any]) -> int:
        request_list = list(n_reordered)
        sched = pos // max(int(len(request_list) / self.batch_schedule), 1)
        cached = self.batch_sizes.get(sched)
        if cached is not None:
            return cached
        if sched > 0 and self.batch_sizes.get(sched - 1) == self.max_batch_size:
            self.batch_sizes[sched] = self.max_batch_size
            return self.max_batch_size
        print("batch_size=auto — detecting largest executable batch", flush=True)
        detected = self._detect_batch_size(request_list, pos)
        self.batch_sizes[sched] = detected
        print(f"Detected batch size: {detected}", flush=True)
        return detected

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        from lm_eval.models.utils import Collator

        pad_and_concat = _import_pad_and_concat()
        res: list[tuple[float, bool]] = []

        def _collate(
            req: tuple[tuple[str, str], list[int], list[int]],
        ) -> tuple[int, tuple[int, ...]]:
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        n_reordered = len(re_ord)

        batch_size: int | str = (
            self.batch_size
            if self.batch_size != "auto"
            else (override_bs if override_bs is not None else 0)
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and n_reordered > 0 and override_bs is None
            else None
        )
        if batch_fn is not None:
            self.batch_sizes = {}

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)

        for chunk in tqdm(chunks, disable=disable_tqdm, desc="Pollux loglikelihood"):
            inps: list[torch.Tensor] = []
            cont_toks_list: list[list[int]] = []
            inplens: list[int] = []
            pad_len: int | None = None

            for _req_pair, ctx_enc, cont_enc in chunk:
                assert ctx_enc and cont_enc
                inp = torch.tensor(
                    (ctx_enc + cont_enc)[-(self.max_length + 1):][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                inplen = int(inp.numel())
                pad_len = max(pad_len or 0, inplen)
                inps.append(inp)
                cont_toks_list.append(cont_enc)
                inplens.append(inplen)

            batched_inps = pad_and_concat(pad_len, inps, padding_side="right")
            multi_logits = F.log_softmax(
                self._model_call(batched_inps).float(), dim=-1
            )

            for (_req_pair, _ctx, cont_toks), logits, inplen in zip(
                chunk, multi_logits, inplens, strict=True
            ):
                contlen = len(cont_toks)
                logits = logits.unsqueeze(0)[:, -contlen:, :]
                cont_tensor = torch.tensor(cont_toks, device=self.device)[None, :]
                greedy = logits.argmax(dim=-1)
                max_equal = bool((greedy == cont_tensor).all().item())
                gathered = torch.gather(logits, 2, cont_tensor.unsqueeze(-1)).squeeze(-1)
                res.append((float(gathered.sum().item()), max_equal))

        return re_ord.get_original(res)

    def loglikelihood_rolling(
        self,
        requests,
        disable_tqdm: bool = False,
    ) -> list[float]:
        from lm_eval import utils as lm_utils

        adaptive: int | None = None
        if self.batch_size == "auto":
            print("batch_size=auto — detecting largest executable batch", flush=True)
            adaptive = self._detect_batch_size()
            print(f"Detected batch size: {adaptive}", flush=True)

        chunk_size = adaptive or (
            self.batch_size if isinstance(self.batch_size, int) else self.max_batch_size
        )
        loglikelihoods: list[float] = []
        for (string,) in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Pollux rolling loglikelihood",
        ):
            token_list = self.tok_encode(string)
            windows = list(
                map(
                    lm_utils.make_disjoint_window,
                    lm_utils.get_rolling_token_windows(
                        token_list=token_list,
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            windows = [(None,) + w for w in windows]
            doc_nll = 0.0
            for start in range(0, len(windows), chunk_size):
                batch = windows[start : start + chunk_size]
                nlls = self._loglikelihood_tokens(
                    batch, disable_tqdm=True, override_bs=len(batch)
                )
                doc_nll += float(sum(nll[0] for nll in nlls))
            loglikelihoods.append(doc_nll)
        return loglikelihoods

    @torch.inference_mode()
    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        outputs: list[str] = []
        for context, gen_kwargs in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Pollux generate_until",
        ):
            assert isinstance(gen_kwargs, dict)
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = int(gen_kwargs.get("max_gen_toks", 256))

            ctx_ids = self.tok_encode(context)
            idx = torch.tensor([ctx_ids], dtype=torch.long, device=self.device)
            generated = list(ctx_ids)

            for _ in range(max_gen_toks):
                logits = self._model_call(idx[:, -self.max_length:])[:, -1, :].float()[0]
                next_id = int(logits.argmax().item())
                generated.append(next_id)
                idx = torch.cat(
                    [idx, torch.tensor([[next_id]], device=self.device, dtype=torch.long)],
                    dim=1,
                )
                decoded = self.tok_decode(generated)
                if any(stop in decoded for stop in until if stop):
                    break
                if next_id == self.eot_token_id:
                    break

            outputs.append(self.tok_decode(generated[len(ctx_ids):]))
        return outputs




# =============================================================================
# Results table — stratified by cognitive category
# =============================================================================

def _primary_metric(metrics: dict[str, Any]) -> tuple[str, float] | None:
    for key in (
        "exact_match,none", "exact,none", "acc_norm,none",
        "acc,none", "f1,none", "mcc,none",
    ):
        if key in metrics and isinstance(metrics[key], (int, float)):
            return key.split(",")[0], float(metrics[key])
    for key, value in metrics.items():
        if key == "alias":
            continue
        if isinstance(value, (int, float)):
            return key.split(",")[0], float(value)
    return None


def _short_task_name(task: str) -> str:
    if task.startswith("blimp_"):
        return task[len("blimp_"):]
    return task


def _parse_score_pct(score: str) -> float | None:
    if not score.endswith("%"):
        return None
    try:
        return float(score[:-1])
    except ValueError:
        return None


def _print_stratified_table(
    results: dict[str, Any],
    checkpoint_name: str,
    structural_tasks: Sequence[str],
    factual_tasks: Sequence[str],
    *,
    fullblimp: bool = False,
) -> None:
    """Print a two-section stratified results table to stdout.

    The structural / factual partition directly tests the paper's central claim:
    that H24 topological quantization geometrically separates fluid intelligence
    (invariant syntactic rules → high structural scores) from crystallised
    intelligence (volatile factual associations → near-random factual scores).
    """
    task_results: dict[str, dict] = results.get("results", {})
    if not task_results:
        print("No task results returned.", flush=True)
        return

    width = 74
    title = f"Pollux evaluation — {checkpoint_name}"
    if fullblimp:
        title += "  [full BLiMP]"

    def _row(task: str) -> tuple[str, str, str]:
        if task not in task_results:
            return task, "—", "skipped"
        primary = _primary_metric(task_results[task])
        if primary is None:
            return task, "—", "—"
        metric, value = primary
        return task, metric, f"{value * 100:.2f}%"

    structural_rows = [_row(t) for t in structural_tasks if t in task_results]
    factual_rows    = [_row(t) for t in factual_tasks    if t in task_results]
    other_tasks     = sorted(
        t for t in task_results
        if t not in set(structural_tasks) | set(factual_tasks)
    )
    other_rows = [_row(t) for t in other_tasks if t in task_results]

    print(f"\n{'=' * width}", flush=True)
    print(title[:width], flush=True)
    print(f"{'=' * width}", flush=True)

    def _section(
        heading: str,
        rows: list[tuple[str, str, str]],
        *,
        compact: bool = False,
    ) -> None:
        if not rows:
            return
        print(f"\n  {heading}", flush=True)
        scores = [
            pct for _, _, score in rows
            if (pct := _parse_score_pct(score)) is not None
        ]
        if scores:
            mean_pct = sum(scores) / len(scores)
            print(
                f"  Summary: {len(scores)} tasks, "
                f"mean {mean_pct:.2f}%  "
                f"(min {min(scores):.2f}%, max {max(scores):.2f}%)",
                flush=True,
            )
        if compact:
            print(
                f"  Per-task scores saved to JSON "
                f"({len(rows)} structural tasks).",
                flush=True,
            )
            return
        print(f"  {'Task':<38} {'Metric':<12} {'Score':>8}", flush=True)
        print(f"  {'-' * (38 + 12 + 9)}", flush=True)
        for task, metric, score in rows:
            label = _short_task_name(task)
            if len(label) > 38:
                label = label[:35] + "..."
            print(f"  {label:<38} {metric:<12} {score:>8}", flush=True)

    _section(
        "STRUCTURAL  (fluid intelligence — expected high)",
        structural_rows,
        compact=fullblimp and len(structural_rows) > 8,
    )
    _section(
        "FACTUAL  (crystallised intelligence — expected near random)",
        factual_rows,
    )
    if other_rows:
        _section("OTHER", other_rows)

    print(f"\n{'=' * width}\n", flush=True)


# =============================================================================
# Checkpoint discovery
# =============================================================================

def _find_checkpoints(ckpt_dir: str) -> list[str]:
    from glob import glob as _glob
    paths = []
    for pat in ("*.plx", "*.pt"):
        paths.extend(
            p for p in _glob(os.path.join(ckpt_dir, pat))
            if not p.endswith(".packed.pt")
        )

    def _key(p: str) -> tuple[int, str]:
        import re
        nums = re.findall(r"\d+", os.path.basename(p))
        return (int(nums[-1]) if nums else -1, p)

    return sorted(dict.fromkeys(paths), key=_key)


def _resolve_path(explicit: str | None, ckpt_dir: str) -> str:
    if explicit:
        if os.path.isfile(explicit):
            return os.path.abspath(explicit)
        candidate = os.path.join(ckpt_dir, explicit)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {explicit!r}")

    paths = _find_checkpoints(ckpt_dir)
    if not paths:
        raise FileNotFoundError(
            f"No .plx or .pt checkpoints found in {ckpt_dir!r}.  "
            "Run pack.py to create a .plx file first."
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

def _parse_batch_size(raw: str | int) -> int | str:
    text = str(raw).strip()
    if text.startswith("auto"):
        return text
    return int(text)


def main() -> None:
    from lm_eval import simple_evaluate

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Pollux checkpoint with lm-evaluation-harness.\n"
            "Results are printed in a two-section table: STRUCTURAL vs FACTUAL."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py\n"
            "  python evaluate.py model.plx\n"
            "  python evaluate.py model.pt --batch-size 32\n"
            "  python evaluate.py model.plx --tasks blimp_wh_island piqa\n"
            "  python evaluate.py model.plx --limit 0.1   # 10% for testing\n"
            "  python evaluate.py model.plx --fullblimp   # 67 BLiMP + 4 factual\n"
        ),
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="",
        help="Path to .plx or .pt checkpoint (interactive wizard if omitted)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK",
        help=(
            "lm-eval task names (aliases supported).  "
            "Defaults to 4 core BLiMP probes + 4 factual tasks, "
            "or all 67 BLiMP tasks when --fullblimp is set."
        ),
    )
    parser.add_argument(
        "--fullblimp",
        action="store_true",
        help="Run all 67 BLiMP tasks instead of the core 4 structural probes.",
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        help='Batch size: integer, "auto", or "auto:N" (default: 16)',
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=16,
        help="Upper bound when --batch-size=auto (default: 16)",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Fraction (0–1) or count of examples per task for quick testing",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Override default few-shot count per task",
    )
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
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="JSON output file (default: eval_results.json)",
    )
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir or os.path.join(str(_HERE), "checkpoints")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    batch_size = _parse_batch_size(args.batch_size)
    eval_tasks, structural_hint = build_eval_tasks(
        explicit_tasks=args.tasks,
        fullblimp=args.fullblimp,
    )

    src = _resolve_path(str(args.checkpoint).strip() or None, ckpt_dir)

    print(f"\nCheckpoint: {src}", flush=True)
    print(f"Device:     {device.type}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    if args.fullblimp and args.tasks is None:
        print("BLiMP:      full suite (67 tasks)", flush=True)
    print(f"Tasks:      {', '.join(eval_tasks)}\n", flush=True)

    model, cfg = load_model(src, device)

    # HuggingFace GPT-2 tokenizer — eos_token_id = 50256, vocab = 50257.
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    except ImportError as exc:
        raise ImportError(
            "transformers is required for evaluate.py.\n"
            "Install it with: pip install transformers"
        ) from exc

    lm = PolluxLM(
        model,
        tokenizer,
        cfg,
        device=device,
        batch_size=batch_size,
        max_batch_size=args.max_batch_size,
    )

    results = simple_evaluate(
        model=lm,
        tasks=eval_tasks,
        batch_size=batch_size,
        max_batch_size=int(args.max_batch_size),
        device=str(device),
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        log_samples=False,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no results.")

    task_results = results.get("results", {})
    structural_tasks = structural_tasks_from_results(task_results, structural_hint)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(task_results, f, indent=2)
    print(f"Saved full results to  {args.output}", flush=True)

    _print_stratified_table(
        results,
        checkpoint_name=os.path.basename(src),
        structural_tasks=structural_tasks,
        factual_tasks=list(_FACTUAL_TASKS),
        fullblimp=args.fullblimp and args.tasks is None,
    )


if __name__ == "__main__":
    main()
