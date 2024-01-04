import numpy as np
import torch

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def dirichlet_noise(x):
    return np.random.dirichlet([.03] * x)


def entropy_fn(log_act_probs):
    return -torch.mean(
        torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
    )
    
def policy_loss_fn(mcts_probs, log_act_probs):
    return -torch.mean(
        torch.sum(mcts_probs*log_act_probs, 1)
    )
    
def kl_divergence(old_probs, new_probs):
    return np.mean(
        np.sum(old_probs * (
            np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)
        ), axis=1)
    )
    
def explained_var(labels, preds):
    return 1 - np.var(labels - preds) / np.var(labels)