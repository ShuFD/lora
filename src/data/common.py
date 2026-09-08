from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Callable


def jsonl(path: str | Path):
    # utf-8-sig accepts JSONL exported by Windows PowerShell as well as plain UTF-8.
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def write_cleaned(input_path: str, output_path: str, normalizer: Callable, max_chars: int) -> dict:
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    seen, kept, skipped = set(), 0, 0
    with output.open("w", encoding="utf-8") as destination:
        for _, row in jsonl(input_path):
            normalized = normalizer(row)
            signature = json.dumps(normalized, ensure_ascii=False, sort_keys=True) if normalized else ""
            if not normalized or len(signature) > max_chars or signature in seen:
                skipped += 1; continue
            seen.add(signature); destination.write(json.dumps(normalized, ensure_ascii=False) + "\n"); kept += 1
    stats = {"input": input_path, "output": str(output), "kept": kept, "skipped": skipped, "max_chars": max_chars}
    output.with_suffix(output.suffix + ".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def parser(description: str):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--input", required=True); result.add_argument("--output", required=True)
    result.add_argument("--max-chars", type=int, default=12000)
    return result
