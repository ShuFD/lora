from src.rl.verifier import run_python_tests


def score(predictions: list[str], tests: list[str], execute: bool = False) -> dict:
    if not execute:
        return {"pass_rate": None, "total": len(predictions), "note": "Code execution disabled; pass --execute-code for trusted local tests."}
    rewards = [run_python_tests(code, test) for code, test in zip(predictions, tests)]
    return {"pass_rate": sum(rewards) / len(rewards) if rewards else 0.0, "passed": int(sum(rewards)), "total": len(rewards)}
