# Pollux — Native $\Lambda_{24}$ Leech-Lattice Transformers

> **Paper:** *Artificial Neural Networks as Discrete Complex Systems: Native Vector-Ternary Training via the Leech Lattice $\Lambda_{24}$*
> Alexander Lavicka · [lavicka@cantab.net](mailto:lavicka@cantab.net)
> Accepted for publication in *Complex & Intelligent Systems* (02/09/2026). This repository accompanies the author's accepted version; the final published version may differ due to copyediting and typesetting.
> Patent: WIPO Application No. PCT/AT2026/060108 and Austrian Patent Application No. A65086/2026 (implementation/software-architecture claims only — see [Licensing](#licensing) below).
> Checkpoints & code archive: Zenodo [10.5281/zenodo.22254791](https://doi.org/10.5281/zenodo.22254791)

---

## At a Glance

Pollux is a research proof-of-concept for training decoder-only LLMs with a transformer backbone that is projected onto the $\Lambda_{24}$ Leech lattice *from the first gradient step*, rather than quantized after the fact. The paper's central question is not whether this beats continuous baselines on a leaderboard, but whether stable training convergence is achievable at all under a **0.76-bit code rate imposed from initialization** — and the empirical answer is yes.

* **0.76 Bits per Parameter:** The transformer backbone (Q/K/V/O projections and MLP layers) is addressed by an 18-bit index into the 196,561-entry $\Lambda_{24}$ codebook (196,560 kissing points + a prepended zero-vector erasure/null attractor), plus a small per-row FP16 RMS scale — 0.75 + ~0.01–0.014 bits/parameter depending on width.
* **Zero-Continuous-Weight Backbone (in principle):** The *observable* backbone weights at any training or inference step are always exact elements of the fixed $\Lambda_{24}$ codebook — never arbitrary floats. Continuous "shadow weights" exist only inside the optimizer as a latent buffer; see [Hardware & Inference Limitations](#hardware--inference-limitations) for what this does and does not mean for today's reference runtime.
* **Parameter-Free Endogenous Kinetics:** No learning-rate schedule, warmup, gradient-clipping threshold, or weight-decay hyperparameter is tuned. Every optimizer constant (momentum, decay, step size, width-inertia) is derived directly from two geometric invariants of $\Lambda_{24}$ — the Voronoi-cell variance $\gamma = G_{24} \approx 0.065771$ and the covering-to-packing ratio $C = \sqrt{2}$. The one required external input is `H_floor`, an empirically measured corpus noise floor (used only as a normalization reference for the training "heat" signal — an information-theoretic analogy, not a literal thermodynamic quantity).
* **A structural/factual asymmetry, not a hallucination cure:** The Voronoi rate-distortion filter is hypothesized (and, at this scale, empirically observed) to bias gradient accumulation toward coherent structural syntax over idiosyncratic factual detail. This **delays and bounds** factual accumulation relative to a continuous baseline at matched token budgets — it does **not** stop the model from generating confident, fluent, and sometimes factually wrong text (see Appendix A of the paper and the qualitative examples below). The design rationale is to pair a syntax-focused core with an external, auditable retrieval store, rather than to eliminate hallucination outright.
* **Small-scale empirical parity:** At the 10k-step ("crystallisation peak") checkpoints, Pollux-1152 and Pollux-1920 reach BLiMP scores comparable to continuously-trained Pythia-160M/410M at similar early token budgets, at a smaller total on-disk footprint. This has been observed at two model scales on one corpus with a single training run each (n = 1) — see [Limitations](#limitations).

---

## The Core Concept

### Why native vector quantization, not post-training compression?
Scalar QAT (e.g. 1.58-bit ternary networks) applies a one-dimensional grid that ignores the geometry of the parameter space and pays an explicit 0.58-bit premium ($\log_2 3 \approx 1.58$ bits) just to represent a reject/erasure symbol. Post-hoc lattice quantization (e.g. Leech-lattice PTQ methods) is applied only after continuous pre-training, so it approximates — and partially distorts — a model that was never optimized under the constraint.

Pollux instead trains **at its final quantization resolution from step zero**: there is no continuous model being approximated. Rather than scalar ternary quantization ($3^{24} \approx 282 \times 10^9$ combinatorial states per 24-dim atom), Pollux addresses the same 24-dimensional atom with the **196,560 kissing points of the Leech lattice $\Lambda_{24}$** — the provably densest sphere packing and optimal vector quantizer in 24 dimensions (Conway & Sloane) — plus a prepended zero vector serving as the erasure/null attractor.

$$
\underbrace{18\,\text{bits}}_{\text{LUT index}} \;/\; \underbrace{24\,\text{params}}_{\text{atom dim}}
\;+\;
\underbrace{16\,\text{bits}}_{\text{FP16 scale}} \;/\; \underbrace{1152\,\text{params}}_{d\text{-dim row}}
\;=\; 0.750 + 0.0138 \approx \mathbf{0.76\;\text{bits/param}}
$$

(For wider models, e.g. $d=1920$, the scale overhead drops to ≈0.0083 bits/param, asymptotically approaching the 0.75-bit geometric floor. 0.76 bits is used throughout as the conservative reference figure.)

### Fluid vs. crystallised intelligence — a filtering hypothesis, tested at small scale
The paper frames SGD noise as an approximately isotropic (AWGN) component superimposed on a directionally coherent structural signal — a standard modeling assumption in the SGD-as-diffusion literature, adopted here as a theoretical motivation rather than something directly measured in these training runs. Under that framing, the $C=\sqrt{2}$ Voronoi deep-hole barrier of $\Lambda_{24}$ acts as a geometric high-pass filter on gradient updates:

* **Fluid intelligence (structural syntax):** Gradients that are directionally coherent across many batches accumulate enough momentum to cross the barrier and commit an atom to a new kissing point.
* **Crystallised intelligence (factual content):** Gradients from sparse, idiosyncratic examples tend to cancel across batches and are pulled back toward the zero-vector erasure attractor by the geometric decay term — but ubiquitous, high-frequency facts can still generate coherent enough gradients to leak through over enough tokens ("kinetic prioritization," not an absolute firewall).

Empirically (Section 4 of the paper), both Pollux configurations show a **bimodal per-task accuracy distribution** on the 67 BLiMP sub-tasks — a compressed "noise zone" (45–55% accuracy) and a larger share of tasks pushed into the statistically-significant "acquired" zone (>55%) than the continuous Pythia baselines at comparable token budgets — consistent with (but not conclusive proof of) the predicted Voronoi-barrier filtering mechanism. See Table 4 in the paper for the full zone breakdown.

This motivates describing Pollux as a candidate **"stateless cognitive architecture"**: a lattice-constrained core intended for fluid syntactic processing, designed to be paired with an external, auditable factual store (retrieval) rather than to serve as both processor and memory. The qualitative generations in Appendix A of the paper illustrate the failure mode this design is meant to address: both checkpoints produce structurally fluent, grammatically well-formed text that regularly invents plausible-sounding but factually incorrect content (e.g. confabulated terms like "endolymphadoproteins"). That confident confabulation is *expected* under this framing, not eliminated by it — grounding still has to come from outside the model.

---

## Empirical Results

Pollux models are evaluated under a **Matched Informational Footprint** protocol: since a $\Lambda_{24}$ parameter carries 0.76 bits versus up to 16 bits for a continuous Pythia parameter, raw parameter counts are not directly comparable, so results are reported by total serialized on-disk footprint. Pythia ([EleutherAI Pythia suite](https://github.com/EleutherAI/pythia)) is used as an **external reference**, not a matched control: it is trained on a different corpus (The Pile vs. our FineWeb-Edu 10B subset) with a different tokenizer. A same-corpus, same-tokenizer baseline is an open item for future work.

| Model | Training tokens | BLiMP (structural) | SciQ (factual) | HellaSwag (factual) | PIQA (factual) | Backbone SRAM | Total on-disk |
|---|---|---|---|---|---|---|---|
| **Pollux-1152** | 2.6B (step 10k) | **69.9%** | 50.3% | 26.4% | 57.7% | **27 MB** | **142 MB** |
| **Pollux-1920** | 2.6B (step 10k) | **73.0%** | 60.7% | 27.2% | 59.8% | **76 MB** | **265 MB** |
| Pythia-160M @ step 2k | 4.2B | 69.7% | 58.7% | 26.9% | 58.4% | 162 MB | 247 MB |
| Pythia-410M @ step 2k | 4.2B | 73.1% | 57.2% | 27.3% | 58.2% | 577 MB | 707 MB |
| Pythia-160M @ Asymptotic | 300B | 73.1% | 72.3% | 29.1% | 61.9% | 162 MB | 247 MB |
| Pythia-410M @ Asymptotic | 300B | 81.9% | 82.4% | 34.5% | 67.2% | 577 MB | 707 MB |

> Random-chance baselines: BLiMP (2-way) = 50.0%; HellaSwag / SciQ (4-way) ≈ 25%; PIQA (2-way) ≈ 50%. HellaSwag and PIQA are near-random for every configuration in this table (best HellaSwag anywhere: 34.5%) and are retained only for baseline continuity; SciQ carries most of the factual-accumulation comparison. All Pollux scores are measured on the packed `.plx` deployment artifact, and Pythia's Comparable-Early-Training row processed ~38% more tokens than Pollux at the same checkpoint (4.2B vs 2.6B) — this is labeled a comparable phase for sample-efficiency framing, not an exact data match.

At comparable early-training exposure, Pollux-1152 and Pollux-1920 reach BLiMP within roughly a point of the size-matched Pythia checkpoint while using a smaller total footprint (38–63% smaller across the two pairs). At Pythia-160M's 300B-token asymptote, Pollux-1920's BLiMP (73.0%) is still comparable (73.1%) — but Pythia's SciQ continues climbing to 72.3–82.4% over that budget while Pollux's factual scores stay near their 10k-step plateau. **Whether that plateau is a hard topological ceiling or just a finite-scale delay that would eventually be breached by enough tokens ("macroscopic entropic leakage") is an open question the paper does not resolve** (Section 4.6).

### Thermodynamic capacity curve
Both configurations show a steep initial loss reduction followed by a plateau; structural (BLiMP) and factual (SciQ/HellaSwag/PIQA) benchmarks stop moving by more than ~1% beyond the 10,000-step ("crystallisation peak") checkpoint, out to the 15,000-step horizon actually tested (~3.9B tokens):

| Checkpoint | Tokens | BLiMP (structural) | SciQ (factual) | HellaSwag (factual) | PIQA (factual) |
|---|---|---|---|---|---|
| **Pollux-1152** | | | | | |
| 5k steps | ~1.3B | 67.5% | 46.5% | 26.6% | 55.7% |
| **10k steps** ⬅ *evaluation checkpoint* | ~2.6B | **69.9%** | **50.3%** | **26.4%** | **57.7%** |
| 15k steps | ~3.9B | 69.9% | 48.4% | 26.6% | 57.7% |
| **Pollux-1920** | | | | | |
| 5k steps | ~1.3B | 72.9% | 56.6% | 26.9% | 58.4% |
| **10k steps** ⬅ *evaluation checkpoint* | ~2.6B | **73.0%** | **60.7%** | **27.2%** | **59.8%** |
| 15k steps | ~3.9B | 73.2% | 61.7% | 27.3% | 60.1% |

The 10,000-step checkpoint is used for all comparative evaluations because neither model improved by more than 1% on any benchmark aggregate beyond it, within the training horizon tested (15k steps / ~3.9B tokens). This is not evidence of a plateau beyond that horizon.

### Held-out validation
Held-out validation loss (a separate FineWeb-Edu 10B subset, disjoint from the training shard) tracks training loss throughout for both configurations with no divergence — consistent with the model generalizing rather than memorizing the training set at this scale, though this has only been checked with a single held-out split per model, not cross-validated.

---

## Limitations

These are carried over directly from the paper's Limitations section (4.6) and apply to everything in this repository, not just the paper text:

* **Single-trajectory runs.** Each configuration (Pollux-1152, Pollux-1920) reflects one training run (n = 1). No multi-seed variance or confidence intervals are reported; agreement across the two architectural scales is a cross-scale check, not a multi-seed robustness check.
* **No matched-footprint scalar-QAT baseline.** There is no same-corpus BitNet-b1.58-style scalar ternary baseline in this repository or the paper. Without it, it is not possible to fully separate effects specific to $\Lambda_{24}$ geometry from the general effect of extreme sub-1-bit quantization on any architecture.
* **No same-corpus/tokenizer Pythia baseline.** The Pythia comparison uses a different corpus (The Pile) and tokenizer than Pollux (FineWeb-Edu 10B, GPT-2 BPE via `tiktoken`), so it is an external reference point, not a controlled ablation.
* **Un-ablated geometric ansatz choices.** The width-inertia factor $\eta_d$ and the momentum coefficients $\beta_1,\beta_2$ (derived from $\gamma$ and $C$) are motivated by, but not uniquely derived from, the lattice geometry, and have not been empirically compared against alternatives (e.g. $\eta_d \equiv 1$, standard Adam defaults) on this architecture. The untied-embeddings design is similarly motivated but not empirically compared against a tied-weight configuration.
* **Open question on the factual plateau.** It is unresolved whether the observed bound on factual accumulation is an unconditional property of the $C=\sqrt{2}$ barrier at any scale, or a finite-scale effect that would eventually be overcome ("macroscopic entropic leakage") with a much larger token budget.
* **Small model / data scale.** Everything here is at the 404M–991M total parameter scale on a 10B-token corpus subset. Broad-coverage benchmarks (MMLU, GPQA) were judged out of scope at this scale; results should not be extrapolated to production-scale models without further validation.
* **Not a demonstrated hardware speedup.** See [Hardware & Inference Limitations](#hardware--inference-limitations) — the current reference runtime does not realize the memory-bandwidth benefit of the 0.76-bit code rate.

---

## Hardware & Inference Limitations

Pollux is a **functional reference implementation for research**, not an optimized production runtime. The following constraints apply to anyone deploying or extending the codebase:

**Packed storage vs. PyTorch runtime:** While the packed `.plx` representation fits entirely in on-chip memory (~27 MB backbone for Pollux-1152), the **current reference PyTorch path materialises dense FP16 weight matrices** at inference time (`PackedH24Linear.materialize()`) for standard `cuBLAS` compatibility. This validates the crystallisation and Iso-Memory *storage* claims and the zero-shot benchmark numbers above, but it does **not** currently deliver SRAM-bound latency or reduced FLOPs — under the present materialisation, inference compute scales with backbone parameter count like any dense model, and it is not matched between Pollux and the Pythia baselines. **Native matrix-free LUT gather–accumulate kernels** (read index → fetch codebook vector → accumulate $\sigma_{\mathrm{rms}} \cdot c$) are required to realize the intended compute/bandwidth benefit, and are future work, not part of this release.

**Edge CPU viability:** Modern CPUs feature large L3 caches (8–32 MB) capable of holding the entire ~9 MB $\Lambda_{24}$ codebook, which is favorable for an index-routing pipeline in principle. By compressing a ~1B-parameter model to a 265 MB on-disk footprint, Pollux is intended to make edge/IoT deployment more tractable where continuous models would trigger out-of-memory failures — but this repository does not yet include an optimized CPU inference kernel to validate that claim end-to-end.

**Architectural strictness:** Custom configurations must satisfy `n_embd % 24 == 0`. Every quantized linear `in_features` must be cleanly divisible by 24 for proper Leech-lattice atom tiling (see Appendix B of the paper for the full set of dimensional/padding constraints, including the 128-divisible vocabulary padding used purely for GPU memory-access alignment).

---

## Repository Structure

```
publish/
│
├── castor.py               # Axiom layer: Leech lattice codebook, constants,
│                           #   nearest-neighbour quantizer, bit-packing.
│                           #   Leaf node — imports nothing from this project.
│
├── pollux.py               # Zero-continuous-weight architecture +
│                           #   parameter-free endogenous-kinetics estimator
│                           #   (pollux_step). Depends only on castor.
│                           #   Contains both training (PolluxH24Linear) and
│                           #   inference (PackedH24Linear) layer classes.
│
├── train.py                # Training entry point. Reads FineWeb-Edu memmap,
│                           #   calls pollux_step, writes .pt checkpoints.
│                           #   No LR schedule, no weight decay, no warmup.
│
├── validate.py              # Computes held-out cross-entropy for a Pollux
│                           #   training checkpoint (.pt) against a reserved,
│                           #   disjoint FineWeb-Edu 10B split. Used to produce
│                           #   the validation-loss curve reported in the paper
│                           #   (Fig. 1) and to check for train/val divergence
│                           #   (a basic memorization check) at any checkpoint.
│
├── prepare_fineweb.py      # Downloads FineWeb-Edu 10B, tokenizes with GPT-2,
│                           #   writes uint16 memmap to data/fineweb_10b.bin.
│
├── pack.py                 # Checkpoint → .plx converter.
│                           #   Quantizes H24 layers to 18-bit LUT indices +
│                           #   FP16 σ_rms per row, INT8-quantized embeddings.
│                           #   Pack at the 10k crystallisation peak checkpoint.
│
├── generate.py             # Text generation from .plx or .pt files.
│                           #   .plx: index_select materialisation + F.linear;
│                           #   native LUT kernels (future) eliminate dense
│                           #   weight-matrix traffic, not FP activations.
│
├── evaluate.py             # lm-eval-harness wrapper. Prints stratified
│                           #   Structural (4 BLiMP) vs Factual (4 MCQ) table.
│                           #   Accepts both .plx and .pt inputs.
│
├── data/                   # Local training corpus (gitignored; created by
│   └── fineweb_10b.bin     #   prepare_fineweb.py)
│
└── checkpoints/            # Training checkpoints (gitignored; written by
    └── pollux_step_*.pt    #   train.py every 2.5k optimizer steps)
```

### Inference & validation pipeline

```
train.py  ──(pollux_10k.pt)──►  pack.py  ──(model.plx)──►  generate.py
                              │                          ──►  evaluate.py
                              └──(pollux_step_*.pt)──►  validate.py  (held-out CE)
```

> **Technical Note on Native Inference:**
> The current reference PyTorch runtime materialises 18-bit indices to FP16 weight tiles via `index_select`, executing via standard `F.linear` / `cuBLAS`. This explicitly validates the zero-shot crystallisation and Iso-Memory *storage* bounds, but does not yet deliver SRAM-bound latency on standard GPUs. True hardware acceleration requires a native C/CUDA/Triton kernel (or dedicated NPU logic) to perform **matrix-free vector scaling**: SRAM lookup of codebook vectors by index, combined with continuous FP16/BF16 activations via scalar–vector multiply–accumulate — eliminating dense $\mathcal{O}(N^2)$ weight-matrix DRAM traffic entirely. This hardware-software isomorphism is discussed in Appendix B of the paper as an engineering target, not a result already achieved.

---

## Quickstart

### 1 — Environment

```bash
conda create -n pollux python=3.11 -y
conda activate pollux
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install tiktoken lm_eval tqdm numpy
# Optional: Triton (highly recommended — massively accelerates the Castor STE
#   projection during training and avoids VRAM bottlenecks during the H24 snap;
#   also speeds checkpoint packing and validate.py's decode pass)
pip install triton
```

### 2 — Run inference from a `.plx` file

```bash
python generate.py model.plx --prompt "The second law of thermodynamics" \
    --max-new-tokens 200 --temperature 0.8 --top-k 50
```

### 3 — Run the evaluation suite

```bash
# Evaluate a packed .plx file (structural vs. factual stratified table)
python evaluate.py model.plx --fullblimp

# Evaluate a raw training checkpoint
python evaluate.py pollux_10k.pt --batch-size 16

# Quick smoke-test (10% of each task)
python evaluate.py model.plx --limit 0.1
```

### 4 — Check held-out validation loss

```bash
# Cross-entropy on a reserved, disjoint FineWeb-Edu 10B split, at any checkpoint
python validate.py checkpoints/pollux_step10000.pt --data data/fineweb_10b_val.bin

# Sweep multiple checkpoints to reproduce the Fig. 1 train/val curve
python validate.py checkpoints/pollux_step*.pt --data data/fineweb_10b_val.bin --csv val_curve.csv
```

`validate.py` reports the standard training-loss moving average alongside the held-out cross-entropy so train/val divergence (a basic memorization check) can be inspected directly; it does not run BLiMP/SciQ/HellaSwag/PIQA — use `evaluate.py` for those.

### 5 — Pack a checkpoint

```bash
# Pack the 10k crystallisation peak checkpoint (recommended)
python pack.py checkpoints/pollux_step10000.pt --output pollux_1152_10k.plx --device cuda
```

### 6 — Minimal Python API

```python
import torch
from pathlib import Path
from pollux import PolluxConfig, PolluxModel

# ── Option A: load from a 0.76-bit .plx packed file ─────────────────────────
# Pollux-1152: ~27.3 MB backbone SRAM; ~142 MB total on disk.
# Weights are materialised from the SRAM codebook via index_select on first use.

from generate import _read_plx   # private .plx reader (standalone, no deps)

device = "cuda" if torch.cuda.is_available() else "cpu"
payload = _read_plx("pollux_1152_10k.plx")
model = PolluxModel.from_packed_checkpoint("pollux_1152_10k.plx", device, payload=payload)
cfg = PolluxConfig.from_dict(payload["config"])
model.eval()

# ── Option B: load from a training checkpoint (.pt) ─────────────────────────
# Observable weights = dynamic Castor H24 projection; continuous pre-weights
# live in optimiser state only.
# ckpt = torch.load("pollux_10k.pt", map_location=device, weights_only=False)
# cfg  = PolluxConfig(**ckpt["config"])
# model = PolluxModel(cfg).to(device)
# model.load_state_dict(ckpt["model"])
# model.eval()

# ── Tokenise and generate ────────────────────────────────────────────────────
import tiktoken
enc = tiktoken.get_encoding("r50k_base")

prompt = "The syntax of a relative clause requires"
ids    = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    out = model.generate(
        ids,
        max_new_tokens=150,
        temperature=0.8,
        top_k=50,
    )

print(enc.decode(out[0].tolist()))
```

---

## Training Data

To download and tokenize the dataset locally, simply run `python prepare_fineweb.py`. This script will stream the 10B token subset, tokenize it, and save the resulting `uint16` binary to `data/fineweb_10b.bin` for fast memmap loading during training. It also writes a disjoint held-out split for use with `validate.py`.

Requires `datasets`, `transformers`, `numpy`, and `tqdm` (in addition to the core training stack). The download is ~20 GB on disk once complete.

---

## Training from Scratch

> **Token Budget Note:** At sequence length 1024, batch size 8, and 32 grad-accum steps, 10,000 steps equal roughly 2.6 billion processed tokens. For larger configurations (e.g., Pollux-1920), training may be executed across multiple sequential resumed runs due to hardware interruptions; optimizer state (including the heat EMA, momentum/variance buffers, and step counter) is fully preserved at each resume point and loss trajectories are stitched by training step.

```bash
# Prepare FineWeb-Edu 10B token shard (creates data/fineweb_10b.bin)
python prepare_fineweb.py

# Train Pollux-1152 (1152-dim, 18 layers, 48 heads — default pollux.py config)
# Targets the 10k crystallisation-peak checkpoint on a single RTX 5090 / ~6 hours
python train.py \
    --target-tokens 9_953_989_333 \
    --wandboff   # remove to enable W&B logging

# Check held-out validation loss at any point during/after training
python validate.py checkpoints/pollux_step10000.pt --data data/fineweb_10b_val.bin

# After ~10k steps, pack the checkpoint
python pack.py checkpoints/pollux_step10000.pt --output pollux_1152_10k.plx
```

### Endogenous Kinetics Calibration (Important)

The optimiser (`pollux_step`) has **no learning-rate schedule, no auxiliary weight decay, gradient clipping, or warmup** — but it does rely on exactly **one environmental boundary condition**: the dataset noise floor `H_floor`. (For the full mathematical derivation of how all other optimiser constants — such as the topological friction $C=\sqrt{2}$ and the Voronoi jitter floor $\gamma = G_{24}$ — are derived strictly from the two $\Lambda_{24}$ axioms, see Section 3.4 of the paper.)

`H_floor` is an **empirical property of the training corpus** — approximately the cross-entropy convergence ceiling of a capacity-matched continuous baseline trained on the same corpus — not an architectural hyperparameter, and it is used only as a normalization reference for the "heat" signal $H(t)$ (an information-theoretic analogy to simulated-annealing temperature, not a literal thermodynamic quantity — see Section 3.4.4 of the paper). For FineWeb-Edu 10B, `DATASET_NOISE_FLOOR = 3.2` in `pollux.py` is anchored at ≈3.2 nats, the reported convergence ceiling for a capacity-matched continuous baseline on this corpus.

**If you train on a different corpus**, measure the continuous-weight convergence ceiling on your data (e.g. with `validate.py` against a short continuous-baseline run), set `H_floor` to that value, and update `DATASET_NOISE_FLOOR` in `pollux.py` before launching `train.py`. A floor set too high underestimates corpus entropy; too low overstates it and distorts the heat normalisation.

---

## Architecture Summary

| Component | Class | Details |
|---|---|---|
| **Training layer** | `PolluxH24Linear` | Forward uses discrete materialised weights; `pollux_step` maintains continuous shadow-weight latents and re-quantizes once per optimizer step |
| **Normalization** | `RMSNorm` | Continuous FP16 learnable gains; magnitude–structure decoupler for the residual stream |
| **Inference layer** | `PackedH24Linear` | Stores `uint8` 18-bit packed indices + `float16` one $\sigma_{\mathrm{rms}}$ per row; `materialize()` expands to FP16 via `codebook.index_select` |
| **Embeddings** | `PackedInt8Embedding` | Per-row INT8 + FP16 scale (kept continuous/high-precision and untied from the LM head — see Section 3.1 of the paper) |
| **LM Head** | `PackedInt8Linear` | Per-row INT8 + FP16 scale (untied: kept at higher precision to avoid large continuous logit updates overwhelming the lattice-filtered backbone gradients) |
| **Optimizer** | `pollux_step` | Heat-modulated Adam-style update with topological friction $1/C$, width-inertia $\eta_d$, geometric reference width $d^*=1152$; straight-through estimator on the discrete projection; no architectural hyperparameters (requires one corpus-specific `H_floor`) |
| **Codebook** | `castor.py` | 196,561 entries (196,560 kissing points + index-0 null/erasure attractor); ~9 MB FP16 |
| **Bit-packing** | `castor.pack_indices` | Bijective 4 × 18-bit → 9-byte; reversible via `unpack_indices` |
| **Validation** | `validate.py` | Held-out cross-entropy over a disjoint FineWeb-Edu 10B split, for any `.pt` checkpoint; used to produce the training/validation curve reported in Fig. 1 of the paper |

---

## Licensing

The source code is released under the **PolyForm Noncommercial License 1.0.0** for academic research, non-commercial experimentation, and scientific reproduction. A copy of the license is available at [https://polyformproject.org/licenses/noncommercial/1.0.0/](https://polyformproject.org/licenses/noncommercial/1.0.0/).

A patent application covering specific **software and hardware-realisation aspects of the reference implementation** is pending:

> **WIPO Application No. PCT/AT2026/060108 and Austrian Patent Application No. A65086/2026**

This application concerns implementation-level engineering and software architecture only. It claims no exclusivity over the underlying mathematical framework, lattice geometry, or dynamical equations described in the paper, which remain fully open for independent scientific replication and use. Commercial utilization, deployment, or hardware integration of the proprietary Pollux reference implementation requires a commercial license from the patent holder. Contact: [lavicka@cantab.net](mailto:lavicka@cantab.net)

---

## Data & Code Availability

Trained checkpoints (`pollux_1152_10k.plx`/`.pt`, `pollux_1920_10k.plx`/`.pt`) are archived on Zenodo: [10.5281/zenodo.22254791](https://doi.org/10.5281/zenodo.22254791). The actively maintained code repository is on GitHub: [https://github.com/alavicka/pollux](https://github.com/alavicka/pollux); a snapshot corresponding to the paper is additionally archived on the same Zenodo record.

---

## Citation

If you use Pollux in your research, please cite:

```bibtex
@article{lavicka2026pollux,
  title   = {Artificial Neural Networks as Discrete Complex Systems: Native Vector-Ternary
             Training via the Leech Lattice $\Lambda_{24}$},
  author  = {Lavicka, Alexander},
  journal = {Complex \& Intelligent Systems},
  year    = {2026},
  note    = {Accepted, author's accepted version. WIPO Patent Application No. PCT/AT2026/060108
             and Austrian Patent Application No. A65086/2026},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6973978}
}
```
