"""Best-effort local code verifier. Never use with untrusted code outside a sandbox."""
from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path


def run_python_tests(code: str, tests: str, timeout_seconds: int = 5) -> float:
    """Return 1.0 only when code plus trusted assertions exits successfully."""
    with tempfile.TemporaryDirectory(prefix="qwen_eval_") as directory:
        target = Path(directory) / "candidate.py"
        target.write_text(code + "\n" + tests, encoding="utf-8")
        try:
            completed = subprocess.run([sys.executable, "-I", str(target)], capture_output=True, text=True, timeout=timeout_seconds, cwd=directory)
            return float(completed.returncode == 0)
        except subprocess.TimeoutExpired:
            return 0.0
