#!/usr/bin/env python
"""
One-stop data downloader for the Qwen2.5-1.5B post-training project.

Downloads three datasets from HuggingFace and writes them as JSONL to data/raw/:

    tatsu-lab/alpaca                      -> data/raw/sft.jsonl          (SFT)
    HuggingFaceH4/ultrafeedback_binarized  -> data/raw/preference.jsonl   (DPO)
    gsm8k ("main", "train" split)          -> data/raw/math.jsonl         (RL/GRPO)

The output JSONL matches the contracts consumed by ``src.data.prepare_*``:

    SFT    : {"instruction", "input", "output"}
    DPO    : {"prompt", "chosen", "rejected"}
    RL/GRPO: {"prompt", "answer"}

Usage
-----
    python scripts/download_data.py                  # download everything, full size
    python scripts/download_data.py --only sft       # just one stage
    python scripts/download_data.py --sft-size 5000  # cap SFT at 5000 rows for a smoke test
    python scripts/download_data.py --force          # re-download even if file already exists

Requires the ``datasets`` package (``pip install datasets>=2.20``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Iterable

from datasets import load_dataset

# Resolve project root (parent of the scripts/ directory).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Per-stage downloaders
# ---------------------------------------------------------------------------

def download_alpaca(output_path: Path, max_rows: int | None) -> int:
    """Download tatsu-lab/alpaca and write instruction/input/output JSONL."""
    print(f"[sft] loading tatsu-lab/alpaca ...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in ds:
            record = {
                "instruction": (row.get("instruction") or "").strip(),
                "input": (row.get("input") or "").strip(),
                "output": (row.get("output") or "").strip(),
            }
            # Skip rows missing required fields; matches prepare_sft filtering.
            if not record["instruction"] or not record["output"]:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(f"[sft] wrote {written} rows -> {output_path}")
    return written


def _last_assistant_text(messages: list[dict] | str) -> str:
    """Extract the final assistant turn from a conversation-style value."""
    if isinstance(messages, str):
        return messages.strip()
    if not messages:
        return ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return (msg.get("content") or "").strip()
    return (messages[-1].get("content") or "").strip()


def download_ultrafeedback(output_path: Path, max_rows: int | None) -> int:
    """Download HuggingFaceH4/ultrafeedback_binarized and write preference pairs."""
    print(f"[dpo] loading HuggingFaceH4/ultrafeedback_binarized (train_prefs) ...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in ds:
            prompt = (row.get("prompt") or "").strip()
            chosen = _last_assistant_text(row.get("chosen"))
            rejected = _last_assistant_text(row.get("rejected"))
            if not prompt or not chosen or not rejected or chosen == rejected:
                continue
            f.write(json.dumps(
                {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                ensure_ascii=False,
            ) + "\n")
            written += 1
    print(f"[dpo] wrote {written} rows -> {output_path}")
    return written


def download_gsm8k(output_path: Path, max_rows: int | None) -> int:
    """Download GSM8K train split (test split is reserved for evaluation)."""
    print(f"[rl] loading gsm8k (main, train) ...")
    ds = load_dataset("gsm8k", "main", split="train")
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            prompt = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not prompt or not answer:
                continue
            f.write(json.dumps(
                {"prompt": prompt, "answer": answer, "id": str(idx)},
                ensure_ascii=False,
            ) + "\n")
            written += 1
    print(f"[rl] wrote {written} rows -> {output_path}")
    return written


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

STAGES: dict[str, tuple[str, Callable[[Path, int | None], int]]] = {
    # stage key : (filename, downloader)
    "sft":         ("sft.jsonl",         download_alpaca),
    "preference":  ("preference.jsonl",  download_ultrafeedback),
    "math":        ("math.jsonl",        download_gsm8k),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SFT / DPO / RL training data into data/raw/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write JSONL files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--only",
        choices=list(STAGES) + ["all"],
        default="all",
        help="Download just one stage or all of them (default: all).",
    )
    parser.add_argument(
        "--sft-size",
        type=int,
        default=None,
        help="Cap SFT rows (default: full ~52k). Useful for smoke tests.",
    )
    parser.add_argument(
        "--dpo-size",
        type=int,
        default=None,
        help="Cap DPO rows (default: full).",
    )
    parser.add_argument(
        "--math-size",
        type=int,
        default=None,
        help="Cap RL/GRPO math rows (default: full GSM8K train = 7473).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target JSONL already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    size_overrides = {"sft": args.sft_size, "preference": args.dpo_size, "math": args.math_size}
    stages = list(STAGES) if args.only == "all" else [args.only]

    summary: list[tuple[str, Path, int]] = []
    for stage in stages:
        filename, downloader = STAGES[stage]
        target = output_dir / filename
        if target.exists() and not args.force:
            with target.open("r", encoding="utf-8") as f:
                existing = sum(1 for _ in f)
            print(f"[{stage}] {target} already exists ({existing} rows); skipping. Use --force to redownload.")
            summary.append((stage, target, existing))
            continue

        try:
            written = downloader(target, size_overrides[stage])
        except Exception as exc:  # noqa: BLE001
            print(f"[{stage}] FAILED: {exc}", file=sys.stderr)
            summary.append((stage, target, 0))
            continue
        summary.append((stage, target, written))

    print("\n=== summary ===")
    for stage, path, count in summary:
        status = "ok" if count > 0 else "FAILED"
        print(f"  {stage:<11} {status:<7} {count:>6} rows  {path}")
    print(f"\nNext step: run the prepare_* scripts, e.g.")
    print(f"  python -m src.data.prepare_sft --input {output_dir/'sft.jsonl'} --output data/processed/sft.jsonl")
    return 0 if all(count > 0 for _, _, count in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
