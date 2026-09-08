def score(predictions: list[str], preferred: list[str]) -> dict:
    """Exact-label evaluation for an externally curated comparison set."""
    correct = sum(pred.strip() == label.strip() for pred, label in zip(predictions, preferred))
    return {"win_rate": correct / len(preferred) if preferred else 0.0, "wins": correct, "total": len(preferred)}
