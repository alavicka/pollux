#!/usr/bin/env python3
# Copyright (c) 2026 Alexander Lavicka.
# This source code is licensed under the PolyForm Noncommercial License 1.0.0.
# A copy of this license is available at https://polyformproject.org/licenses/noncommercial/1.0.0/
# Commercial utilization or hardware integration requires a separate license from the patent holder.
"""Download FineWeb-Edu sample-10BT, tokenize with GPT-2, write local uint16 memmap binary."""

from __future__ import annotations

import argparse
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_SUBSET = "sample-10BT"
DEFAULT_OUTPUT = "data/fineweb_10b.bin"
DEFAULT_TOKENIZER = "gpt2"
DEFAULT_TEXT_BATCH = 256
FLUSH_TOKEN_THRESHOLD = 1_000_000


def _repo_root() -> str:
    return os.path.abspath(os.path.dirname(__file__))


def _encode_text_batch(
    tokenizer: AutoTokenizer,
    texts: list[str],
    *,
    active_vocab: int,
    eos_id: int,
) -> list[int]:
    tokens: list[int] = []
    for text in texts:
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        for token in ids:
            value = int(token)
            if 0 <= value < active_vocab:
                tokens.append(value)
        tokens.append(int(eos_id))
    return tokens


def _flush_tokens(handle, token_buffer: list[int], *, threshold: int = FLUSH_TOKEN_THRESHOLD) -> None:
    while len(token_buffer) >= threshold:
        chunk = np.asarray(token_buffer[:threshold], dtype=np.uint16)
        chunk.tofile(handle)
        del token_buffer[:threshold]


def prepare_fineweb_bin(
    *,
    output_path: str,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    text_batch_size: int = DEFAULT_TEXT_BATCH,
    subset: str = FINEWEB_SUBSET,
) -> int:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")
    eos_id = int(tokenizer.eos_token_id)
    active_vocab = int(tokenizer.vocab_size)

    dataset = load_dataset(FINEWEB_DATASET, name=subset, split="train", streaming=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    token_buffer: list[int] = []
    docs_seen = 0
    with open(output_path, "wb") as handle:
        text_batch: list[str] = []
        progress = tqdm(dataset, desc=f"Tokenizing {FINEWEB_DATASET}/{subset}", unit="doc")
        for sample in progress:
            text = str(sample.get("text", "")).strip()
            if not text:
                continue
            text_batch.append(text)
            docs_seen += 1
            if len(text_batch) < int(text_batch_size):
                continue
            token_buffer.extend(
                _encode_text_batch(
                    tokenizer,
                    text_batch,
                    active_vocab=active_vocab,
                    eos_id=eos_id,
                )
            )
            text_batch.clear()
            _flush_tokens(handle, token_buffer)
            progress.set_postfix(tokens=f"{len(token_buffer):,}", docs=f"{docs_seen:,}")

        if text_batch:
            token_buffer.extend(
                _encode_text_batch(
                    tokenizer,
                    text_batch,
                    active_vocab=active_vocab,
                    eos_id=eos_id,
                )
            )
        if token_buffer:
            np.asarray(token_buffer, dtype=np.uint16).tofile(handle)

    total_tokens = int(os.path.getsize(output_path) // np.dtype(np.uint16).itemsize)
    print(
        f"Wrote {total_tokens:,} tokens ({os.path.getsize(output_path) / (1024 ** 3):.2f} GiB) "
        f"from {docs_seen:,} documents -> {output_path}",
        flush=True,
    )
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local FineWeb-Edu uint16 token binary")
    parser.add_argument(
        "--output",
        default=os.path.join(_repo_root(), DEFAULT_OUTPUT),
        help=f"Output path (default: {_repo_root()}/{DEFAULT_OUTPUT})",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="HuggingFace tokenizer id/path")
    parser.add_argument(
        "--text-batch",
        type=int,
        default=DEFAULT_TEXT_BATCH,
        help="Number of documents tokenized per batch",
    )
    parser.add_argument("--subset", default=FINEWEB_SUBSET, help="FineWeb-Edu subset name")
    args = parser.parse_args()
    prepare_fineweb_bin(
        output_path=str(args.output),
        tokenizer_name=str(args.tokenizer),
        text_batch_size=int(args.text_batch),
        subset=str(args.subset),
    )


if __name__ == "__main__":
    main()
