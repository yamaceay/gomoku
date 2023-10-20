import numpy as np

import torch

def softmax(x: np.ndarray) -> np.ndarray:
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def dirichlet_noise(x: int):
    return np.random.dirichlet([.03] * x)


def entropy_fn(log_act_probs: torch.Tensor) -> torch.Tensor:
    return -torch.mean(
        torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
    )
    
def policy_loss_fn(mcts_probs: torch.Tensor, log_act_probs: torch.Tensor) -> torch.Tensor:
    return -torch.mean(
        torch.sum(mcts_probs*log_act_probs, 1)
    )
    
def kl_divergence(old_probs: np.ndarray, new_probs: np.ndarray) -> float:
    return np.mean(
        np.sum(old_probs * (
            np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)
        ), axis=1)
    )
    
def explained_var(labels: np.ndarray, preds: np.ndarray) -> float:
    return 1 - np.var(labels - preds) / np.var(labels)