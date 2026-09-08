import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta: float = 0.1):
    """Reference DPO loss useful for unit tests and diagnostics."""
    logits = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    return -F.logsigmoid(logits).mean(), logits.detach()
