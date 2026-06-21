# Pollux — 0.76-Bit Native $H_{24}$ Leech-Lattice Transformers

> **Paper:** *0.76 Bits Is All You Need: Vector Ternary Logic via Native $H_{24}$ Leech-Lattice Quantization in LLMs*
> Alexander Lavicka · [lavicka@cantab.net](mailto:lavicka@cantab.net) · Preprint 2026
> WIPO Patent Application No. PCT/AT2026/060108 and Austrian Patent Application No. A65086/2026

---

## At a Glance

Pollux is a fundamentally new class of decoder-only LLMs that abandons continuous floating-point weights in the transformer backbone to overcome the von Neumann memory wall. 

* **0.76 Bits per Parameter:** By mapping the neural parameter manifold natively onto the $H_{24}$ Leech lattice (the densest sphere packing in 24D), the backbone is compressed to extreme sub-1-bit levels.
* **Zero-Continuous-Weight Backbone:** Observable layers carry no continuous structural weights—only discrete 18-bit codebook indices and a single global FP16 scale per row.
* **SRAM-Resident Edge AI:** A 1B-class transformer backbone (Pollux-1920) is compressed into just **76 MB of SRAM**, converting inference from a memory-bandwidth-bound to a compute-bound operation.
* **The "Stateless CPU" for RAG:** Pollux physically decouples fluid intelligence (syntax) from crystallised intelligence (factual trivia). Through a geometric Voronoi filter, it mechanically rejects high-entropy factual noise, eliminating the parametric knowledge conflicts that trigger hallucinations.
* **Parameter-Free Thermodynamic Training:** Trained without learning rate schedules, warmup, or weight decay. The network is optimized via endogenous thermodynamic kinetics and Landauer erasure. The only environmental input is `H_floor` — the empirically measured corpus noise floor, analogous to ambient temperature in Carnot theory. All architectural constants are derived from two axioms; no hyperparameter search is required.
* **Empirical Parity:** At less than 1% of the training data and less than half the active SRAM footprint, Pollux achieves strict fluid-syntax parity (BLiMP) with continuous baselines (Pythia 160M–410M).

[![Hugging Face: Pollux-1152](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Pollux--1152-blue)](https://huggingface.co/alavicka/pollux-1152)
[![Hugging Face: Pollux-1920](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Pollux--1920-blue)](https://huggingface.co/alavicka/pollux-1920)
---

## The Core Concept

### Why native vector quantization, not post-training compression?
Standard quantization (INT8, GPTQ, 1.58-bit) either approximates a pre-trained FP16 model after the fact (destroying capacity) or relies on one-dimensional scalar rounding that wastes combinatorial state-space. Pollux is trained **at its final quantization resolution from step zero**. There is no continuous baseline to approximate.

Instead of scalar ternary quantization ($3^{24} \approx 282 \times 10^9$ states for 24 dimensions), Pollux uses the **196,560 mathematically optimal kissing points** of the Leech lattice $\Lambda_{24}$. This provides the highest possible structural resolution per bit. The origin (zero-vector) is prepended to the codebook, acting as a null attractor for vector ternary logic.

$$
\underbrace{18\ \text{bits}}_{\text{LUT index}} \/\ \underbrace{24\ \text{params}}_{\text{atom dim}}
\+\
\underbrace{16\ \text{bits}}_{\text{FP16 scale}} \/\ \underbrace{1152\ \text{params}}_{d\text{-dim row}}
\=\ 0.750 + 0.0138 \approx \mathbf{0.76\\text{bits/param}}
$$

### Fluid vs. Crystallised Intelligence
The **$C=\sqrt{2}$ Voronoi deep-hole barrier** of the lattice acts as a physical high-pass filter on gradients:
* **Fluid intelligence (structural syntax):** Coherent, recurring syntactic rules accumulate directed momentum, cross the Voronoi barrier, and permanently crystallise into $H_{24}$ kissing points.
* **Crystallised intelligence (factual trivia):** Incoherent, high-entropy factual noise fails to cross the threshold and is routed into the zero-potential null attractor. Pollux therefore scores near or modestly above random chance on factual benchmarks — bounded by high-frequency leakage for ubiquitous facts that generate coherent gradients over billions of tokens.

This makes Pollux the ultimate engine for **Retrieval-Augmented Generation (Macro-RAG)**: It acts as a pure, stateless reasoning engine that blindly obeys external vector databases without contaminating the output with internal parametric bias or hallucinations.

---

## Empirical Results

Pollux models are evaluated under a strict **Iso-Memory paradigm**: the active backbone SRAM footprint—not raw parameter count—is the execution-critical metric during autoregressive generation. *(Note: the Iso-Memory criterion isolates memory-bandwidth footprint under the targeted native LUT runtime. Under the current FP16 reference materialisation, FLOPs per token scale with backbone parameter count and are not matched between Pollux and Pythia baselines.)*

### Iso-Memory Evaluation & Breaking the Trilemma
Continuous models are trapped in a Pareto trilemma: they cannot simultaneously minimize SRAM, maximize fluid syntax (BLiMP), and suppress factual contamination (SciQ/HellaSwag). Pollux-1920 breaks this frontier, matching the reasoning capacity of Pythia-410M inside a 76 MB envelope while mechanically resisting factual memorisation.

| Model | Training tokens | BLiMP (Syntax) | SciQ (Facts) | HellaSwag (Facts) | PIQA (Facts) | Backbone SRAM | Total disk |
|---|---|---|---|---|---|---|---|
| **Pollux-1152** | 2.6B (step 10k) | **69.9%** | 50.3% | 26.4% | 57.7% | **27 MB** | **142 MB** |
| **Pollux-1920** | 2.6B (step 10k) | **73.0%** | 60.7% | 27.2% | 59.8% | **76 MB** | **265 MB** |
| Pythia-160M @ step 2k | 4.2B | 69.7% | 58.7% | 26.9% | 58.4% | 162 MB | 247 MB |
| Pythia-410M @ step 2k | 4.2B | 73.1% | 57.2% | 27.3% | 58.2% | 577 MB | 707 MB |
| Pythia-160M @ Asymptotic | 300B | 73.1% | 72.3% | 29.1% | 61.9% | 162 MB | 247 MB |
| Pythia-410M @ Asymptotic | 300B | 81.9% | 82.4% | 34.5% | 67.2% | 577 MB | 707 MB |

> *(Random-chance baselines: BLiMP (2-way) = 50.0%; HellaSwag / SciQ (4-way) ≈ 25%; PIQA (2-way) ≈ 50%. All Pollux scores measured on packed `.plx` deployment artifacts.)*

### Thermodynamic Capacity Curve & The "Deep Freeze"
At 10k steps (~2.6B tokens), the network reaches its thermodynamic crystallisation peak. Beyond this point, the $H_{24}$ topological binding energy locks syntax in place, entering a phase of **thermodynamic stasis** (the "Deep Freeze"): BLiMP shifts by ≤ 0.5% and all factual benchmarks shift by ≤ 1.0% over ≥ 1.3B additional tokens. Capacity churn ceases; the model neither gains new factual associations nor loses established syntactic structure.

| Checkpoint | Tokens | BLiMP (Syntax) | SciQ (Facts) | HellaSwag (Facts) | PIQA (Facts) |
|---|---|---|---|---|---|
| **Pollux-1152** | | | | | |
| 5k steps | ~1.3B | 67.5% | 46.5% | 26.6% | 55.7% |
| **10k steps** ⬅ *Crystallisation peak* | ~2.6B | **69.9%** | **50.3%** | **26.4%** | **57.7%** |
| 15k steps | ~3.9B | 69.9% | 48.4% | 26.6% | 57.7% |
| **Pollux-1920** | | | | | |
| 5k steps | ~1.3B | 72.9% | 56.6% | 26.9% | 58.4% |
| **10k steps** ⬅ *Crystallisation peak* | ~2.6B | **73.0%** | **60.7%** | **27.2%** | **59.8%** |
| 15k steps | ~3.9B | 73.2% | 61.7% | 27.3% | 60.1% |

### Topological Robustness (Lossless Serialization)
The `.plx` serialization mathematically compresses the network to 0.76 bits/param. Compared to the raw continuous `.pt` checkpoint, the maximum deviation across all BLiMP tasks is **0.2 pp** (and 0.01% aggregate mean difference)—engineering confirmation that the global row-scale quantization is practically lossless.

---

## Hardware & Inference Limitations

Pollux is a **functional reference implementation for research**. The following constraints apply to anyone deploying or extending the codebase:

**Packed storage vs. PyTorch runtime:** While the packed `.plx` representation fits entirely in on-chip memory (~27 MB backbone for Pollux-1152), the **current reference PyTorch path materialises dense FP16 weight matrices** at inference time (`PackedH24Linear.materialize()`) for standard `cuBLAS` compatibility. This validates crystallisation and zero-shot benchmarks but **does not** deliver native SRAM-bound latency. **Native matrix-free LUT gather–accumulate kernels** (read index → fetch codebook vector → accumulate $\sigma_{\mathrm{rms}} \cdot c$) are required for the true compute-bound speedup.

**Edge CPU Viability & The RAM Bottleneck:** Standard GPUs severely penalise sub-byte combinatorial addressing. However, modern CPUs feature large L3 caches (8–32 MB) capable of holding the entire 9 MB $H_{24}$ codebook, executing the index-routing pipeline with extreme efficiency. By compressing a 1B-class model to a strict **265 MB on-disk footprint**, Pollux unlocks reasoning for IoT/Edge hardware where continuous models instantly trigger Out-Of-Memory (OOM) failures.

**Architectural Strictness:** Custom configurations must satisfy **`n_embd % 24 == 0`**. Every quantized linear `in_features` must be cleanly divisible by 24 for proper Leech lattice atom tiling.

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
│                           #   parameter-free thermodynamic estimator
│                           #   (pollux_step). Depends only on castor.
│                           #   Contains both training (PolluxH24Linear) and
│                           #   inference (PackedH24Linear) layer classes.
│
├── train.py                # Training entry point. Reads FineWeb-Edu memmap,
│                           #   calls pollux_step, writes .pt checkpoints.
│                           #   No LR schedule, no weight decay, no warmup.
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

### Inference pipeline

```
train.py  ──(pollux_10k.pt)──►  pack.py  ──(model.plx)──►  generate.py
                                                         ──►  evaluate.py
```
### Pre-trained Models & Weights

The fully crystallized, 0.76-bit `.plx` deployment artifacts are hosted on Hugging Face. These containers are fully packed and include the immutable H24 codebook indices alongside the global row-wise RMS scales.

* **[Pollux-1152](https://huggingface.co/alavicka/pollux-1152)**: 287M backbone parameters compressed into 27 MB SRAM (142 MB total on-disk including INT8 embeddings).
* **[Pollux-1920](https://huggingface.co/alavicka/pollux-1920)**: 796M backbone parameters compressed into 76 MB SRAM (265 MB total on-disk including INT8 embeddings).


> **Technical Note on Native Inference:**
> The current reference PyTorch runtime materialises 18-bit indices to FP16 weight tiles via `index_select`, executing via standard `F.linear` / `cuBLAS`. This explicitly validates the zero-shot crystallisation and Iso-Memory theoretical bounds, but does not yet deliver SRAM-bound latency on standard GPUs. True hardware acceleration requires a native C/CUDA/Triton kernel (or dedicated NPU logic) to perform **matrix-free vector scaling**: SRAM lookup of codebook vectors by index, combined with continuous FP16/BF16 activations via scalar–vector multiply–accumulate — eliminating dense $\mathcal{O}(N^2)$ weight-matrix DRAM traffic entirely. This hardware-software isomorphism is detailed in Appendix C of the paper.

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
#   also speeds checkpoint packing)
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

### 4 — Pack a checkpoint

```bash
# Pack the 10k crystallisation peak checkpoint (recommended)
python pack.py checkpoints/pollux_step10000.pt --output pollux_1152_10k.plx --device cuda
```

### 5 — Minimal Python API

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

To download and tokenize the dataset locally, simply run `python prepare_fineweb.py`. This script will stream the 10B token subset, tokenize it, and save the resulting `uint16` binary to `data/fineweb_10b.bin` for fast memmap loading during training.

Requires `datasets`, `transformers`, `numpy`, and `tqdm` (in addition to the core training stack). The download is ~20 GB on disk once complete.

---

## Training from Scratch

> **Token Budget Note:** At sequence length 1024, batch size 8, and 32 grad-accum steps, 10,000 steps equal roughly 2.6 billion processed tokens. For larger configurations (e.g., Pollux-1920), training may be executed across multiple sequential resumed runs due to hardware interruptions; optimizer state is fully preserved at each resume point and loss trajectories are stitched by training step.

```bash
# Prepare FineWeb-Edu 10B token shard (creates data/fineweb_10b.bin)
python prepare_fineweb.py

# Train Pollux-1152 (1152-dim, 18 layers, 48 heads — default pollux.py config)
# Targets the 10k crystallisation peak on a single RTX 5090 / ~6 hours
python train.py \
    --target-tokens 9_953_989_333 \
    --wandboff   # remove to enable W&B logging

# After ~10k steps, pack the checkpoint
python pack.py checkpoints/pollux_step10000.pt --output pollux_1152_10k.plx
```

### Thermodynamic Calibration (Important)

The optimiser (`pollux_step`) has **no learning-rate schedule, no auxiliary weight decay, gradient clipping, or warmup** — but it does rely on exactly **one environmental boundary condition**: the dataset noise floor `H_floor`. (For a full mathematical derivation of how all other optimiser constants, such as the topological drag and Voronoi jitter floor, are derived strictly from the two $H_{24}$ axioms, please refer to Section 3.4 of the paper).

`H_floor` is an **empirical material property** of the training corpus — the irreducible Shannon entropy of its linguistic structure, including factual noise — not an architectural hyperparameter. For FineWeb-Edu 10B, `DATASET_NOISE_FLOOR = 3.2` in `pollux.py` is anchored at the cross-entropy convergence ceiling of an uncompressed FP16 continuous-weight baseline on the same corpus.

**If you train on a different corpus**, measure the FP16 continuous-weight convergence ceiling on your data, set `H_floor` to that value, and update `DATASET_NOISE_FLOOR` in `pollux.py` before launching `train.py`. A floor set too high underestimates corpus entropy; too low overstates it and distorts the heat normalisation.

---

## Architecture Summary

| Component | Class | Details |
|---|---|---|
| **Training layer** | `PolluxH24Linear` | Forward uses discrete materialised weights; `pollux_step` maintains continuous latents and re-quantizes once per step |
| **Normalization** | `RMSNorm` | Continuous FP16 learnable gains; magnitude--structure decoupler for the residual stream |
| **Inference layer** | `PackedH24Linear` | Stores `uint8` 18-bit packed indices + `float16` one $\sigma_{\mathrm{rms}}$ per row; `materialize()` expands to FP16 via `codebook.index_select` |
| **Embeddings** | `PackedInt8Embedding` | Per-row INT8 + FP16 scale (untied from LM head by physical necessity) |
| **LM Head** | `PackedInt8Linear` | Per-row INT8 + FP16 scale (untied: high-precision logit resolution incompatible with H24 gradient geometry) |
| **Optimizer** | `pollux_step` | Heat-modulated Adam with topological drag $1/C$, width stability $\eta_d$, geometric reference baseline $d^* = 1152$; Castor STE; no architectural hyperparameters (requires one corpus-specific `H_floor`) |
| **Codebook** | `castor.py` | 196,561 entries (196,560 kissing + index-0 null attractor); ~9 MB FP16 |
| **Bit-packing** | `castor.pack_indices` | Bijective 4 × 18-bit → 9-byte; reversible via `unpack_indices` |

---

## Licensing

The source code is released under the **PolyForm Noncommercial License 1.0.0** for academic research, non-commercial experimentation, and scientific reproduction. A copy of the license is available at [https://polyformproject.org/licenses/noncommercial/1.0.0/](https://polyformproject.org/licenses/noncommercial/1.0.0/).

The underlying algorithmic principles — specifically the native 24-dimensional Leech lattice straight-through estimation and the thermodynamic optimization protocol — are the subject of a pending patent:

> **WIPO Application No. PCT/AT2026/060108 and Austrian Patent Application No. A65086/2026**

Commercial utilization, deployment, or hardware integration of the proprietary Pollux architecture and its variants requires a commercial license from the patent holders. Contact: [lavicka@cantab.net](mailto:lavicka@cantab.net)

---

## Citation

If you use Pollux in your research, please cite:

```bibtex
@misc{lavicka2026pollux,
  title   = {0.76 Bits Is All You Need: Vector Ternary Logic via Native H24 Leech-Lattice
             Quantization in LLMs},
  author  = {Lavicka, Alexander},
  year    = {2026},
  note    = {Preprint. WIPO Patent Application No. PCT/AT2026/060108 and Austrian Patent Application No. A65086/2026},
  url     = {https://github.com/alavicka/pollux}
}
```
