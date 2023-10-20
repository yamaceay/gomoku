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

        self.policy_layers = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*self.M*self.N, self.M*self.N),
            nn.LogSoftmax(dim=1)
        )

        self.value_layers = nn.Sequential(
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
        x_policy = self.policy_layers(x)
        x_value = self.value_layers(x)
        return x_policy, x_value

class Zero_Net():
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
            self.load_model(model_file)

    def forward_batch(self, state_batch: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        state_batch = self.torch_batch(state_batch)
        log_act_probs, value = self.cnn(state_batch)
        log_act_probs, value = log_act_probs.detach(), value.detach()
        act_probs = np.exp(log_act_probs.cpu().numpy())
        return act_probs, value.cpu().numpy()

    def forward(self, state: Gomoku) -> tuple[list[tuple[float, int]], float]:
        actions = sorted(state.actions())
        legal_positions = [a[0] * state.N + a[1] for a in actions]
        curr_state = state.encode().reshape(
                -1, 4, self.M, self.N)
        curr_state = self.torch_batch(curr_state)
        log_act_probs, value = self.cnn(curr_state)
        log_act_probs, value = log_act_probs.detach(), value.detach()
        act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value[0][0]
    
    def predict(self, state: Gomoku) -> tuple[list[tuple[float, tuple[int, int]]], float]:
        with torch.no_grad():
            act_probs, value = self.forward(state)
        act_probs = sortfn([(p, (a // state.N, a % state.N)) for a, p in act_probs])
        return act_probs, value

    def fit_one(self, batch: list, gamma: float = 1.0) -> tuple[float, float]:
        states, policies, rewards, *next_states = batch
        next_state_given = len(next_states)
        
        states = self.torch_batch(states)
        policies = self.torch_batch(policies)
        rewards = self.torch_batch(rewards)

        self.optimizer.zero_grad()

        log_act_probs, value = self.cnn(states)
        if next_state_given:
            next_states = self.torch_batch(next_states[0])
            _, next_value = self.cnn(next_states)
            rewards += gamma * next_value.view(-1).detach()

        value_loss = F.mse_loss(value.view(-1), rewards)
        policy_loss = policy_loss_fn(policies, log_act_probs)
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        entropy = entropy_fn(log_act_probs)
        return loss.item(), entropy.item()
    
    def torch_batch(self, batch: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(np.ascontiguousarray(batch)).to(self.device)

    def load_model(self, model_file: str):
        net_params = torch.load(model_file)
        self.cnn.load_state_dict(net_params)

    def save_model(self, model_file: str):
        net_params = self.cnn.state_dict()
        torch.save(net_params, model_file)

# if __name__ == '__main__':
#     from torchsummary import summary
#     from torchviz import make_dot
    
#     (M, N, K) = (8, 8, 5)
#     net = CNN(M, N)
#     summary(net, (4, M, N))
#     x = torch.rand(1, 4, M, N)
    
#     graph = make_dot(net(x), params=dict(net.named_parameters()))
#     graph.render(f"out/{M}_{N}_{K}/cnn", format="png")
    