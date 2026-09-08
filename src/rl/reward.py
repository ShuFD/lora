from __future__ import annotations
import re


def extract_answer(text: str) -> str:
    """Prefer \boxed{}, then final-answer markers, then the final number/token."""
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed: return boxed[-1].strip()
    marker = re.findall(r"(?:final answer|answer)\s*[:：]\s*([^\n]+)", text, flags=re.I)
    if marker: return marker[-1].strip().rstrip(".")
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
    return numbers[-1] if numbers else ""


def normalized(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).lower().replace(",", "")


def math_reward(completion: str, answer: str, max_chars: int = 4000) -> float:
    if len(completion) > max_chars: return 0.0
    return float(normalized(extract_answer(completion)) == normalized(answer))


def make_math_reward(answers: list[str]):
    index = {str(i): answer for i, answer in enumerate(answers)}
    def reward(completions, **kwargs):
        expected = kwargs.get("answer", [""] * len(completions))
        return [math_reward(c[0]["content"] if isinstance(c, list) else c, a) for c, a in zip(completions, expected)]
    return reward
