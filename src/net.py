import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .calc import policy_loss_fn, entropy_fn
from .gomoku import Gomoku, sortfn

class CNN(nn.Module):
    def __init__(self, M: int, N: int):
        super().__init__()

        self.M = M
        self.N = N

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.act_layers = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*self.M*self.N, self.M*self.N),
            nn.LogSoftmax(dim=1)
        )

        self.val_layers = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2*self.M*self.N, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, state_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_layers(state_input)
        x_act = self.act_layers(x)
        x_val = self.val_layers(x)
        return x_act, x_val

class Policy_Value_Net():
    def __init__(self, 
                 game_kwargs: tuple[int, int, int],
                 model_file: str = None, 
                 device: torch.DeviceObjType = torch.device('cpu'),
                 opt_args: dict = {}):
        self.device = device
        self.M, self.N, _ = game_kwargs
        self.opt_args = opt_args

        self.cnn = CNN(self.M, self.N).to(device)
        self.optimizer = optim.Adam(self.cnn.parameters(), **opt_args)

        if model_file:
            net_params = torch.load(model_file)
            self.cnn.load_state_dict(net_params)

    def forward(self, state_batch: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        state_batch = self.torch_batch(state_batch)
        log_act_probs, value = self.cnn(state_batch)
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.detach().cpu().numpy()

    def policy_value_fn(self, state: Gomoku) -> tuple[list[tuple[float, int]], float]:
        actions = sorted(state.actions())
        legal_positions = [a[0] * state.N + a[1] for a in actions]
        current_state = state.encode().reshape(
                -1, 4, self.M, self.N)
        current_state = self.torch_batch(current_state)
        log_act_probs, value = self.cnn(current_state)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.detach()[0][0]
        return act_probs, value
    
    def policy_value_fn_sorted(self, state: Gomoku) -> tuple[list[tuple[float, tuple[int, int]]], float]:
        act_probs, value = self.policy_value_fn(state)
        act_probs = sortfn([(p, (a // state.N, a % state.N)) for a, p in act_probs])
        return act_probs, value

    def fit_one(self, batch: list, gamma: float = 1.0) -> tuple[float, float]:
        state_batch, mcts_probs, winner_batch, *next_state_batch = batch
        next_state_given = len(next_state_batch)
        
        state_batch = self.torch_batch(state_batch)
        mcts_probs = self.torch_batch(mcts_probs)
        winner_batch = self.torch_batch(winner_batch)

        self.optimizer.zero_grad()

        log_act_probs, value = self.cnn(state_batch)
        if next_state_given:
            next_state_batch = next_state_batch[0]
            next_state_batch = self.torch_batch(next_state_batch)
            _, next_value = self.cnn(next_state_batch)
            winner_batch += gamma * next_value.view(-1).detach()

        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = policy_loss_fn(mcts_probs, log_act_probs)
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        entropy = entropy_fn(log_act_probs)
        return loss.item(), entropy.item()
    
    def torch_batch(self, batch: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(np.ascontiguousarray(batch)).to(self.device)

    def save_model(self, model_file: str):
        net_params = self.cnn.state_dict()
        torch.save(net_params, model_file)
