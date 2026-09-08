from src.rl.reward import math_reward


def score(predictions: list[str], answers: list[str]) -> dict:
    rewards = [math_reward(prediction, answer) for prediction, answer in zip(predictions, answers)]
    return {"accuracy": sum(rewards) / len(rewards) if rewards else 0.0, "correct": int(sum(rewards)), "total": len(rewards)}
