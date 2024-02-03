import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .calc import policy_loss_fn, entropy_fn
from .gomoku import Gomoku, sortfn

def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# class CNN(nn.Module):
#     def __init__(self, M: int, N: int):
#         super().__init__()

#         self.M = M
#         self.N = N

#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(4, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU()
#         )

#         self.act_layers = nn.Sequential(
#             nn.Conv2d(128, 4, kernel_size=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(4*self.M*self.N, self.M*self.N),
#             nn.LogSoftmax(dim=1)
#         )

#         self.val_layers = nn.Sequential(
#             nn.Conv2d(128, 2, kernel_size=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(2*self.M*self.N, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Tanh()
#         )

#     def forward(self, state_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         x = self.conv_layers(state_input)
#         x_act = self.act_layers(x)
#         x_val = self.val_layers(x)
#         return x_act, x_val

class CNN(nn.Module):
    def __init__(self, M: int, N: int):
        super(CNN, self).__init__()

        self.M = M
        self.N = N
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*self.M*self.N,
                                 self.M*self.N)

        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*self.M*self.N, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.M*self.N)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.M*self.N)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val

class Policy_Value_Net():
    def __init__(self, 
                 game_kwargs: tuple[int, int, int],
                 model_file: str = None, 
                 device: torch.DeviceObjType = torch.device('cpu')):
        self.device = device
        self.M = game_kwargs[0]
        self.N = game_kwargs[1]
        self.l2_const = 1e-4

        self.policy_value_net = CNN(self.M, self.N).to(device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        state_batch = np.ascontiguousarray(state_batch)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()

    def policy_value_fn(self, state: Gomoku) -> tuple[list[tuple[float, int]], float]:
        actions = sorted(state.actions())
        legal_positions = [a[0] * state.N + a[1] for a in actions]
        current_state = np.ascontiguousarray(state.encode().reshape(
                -1, 4, self.M, self.N))
        log_act_probs, value = self.policy_value_net(
                torch.from_numpy(current_state).to(self.device).float())
        act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value
    
    def policy_value_fn_sorted(self, state: Gomoku) -> tuple[list[tuple[float, tuple[int, int]]], float]:
        act_probs, value = self.policy_value_fn(state)
        act_probs = sortfn([(p, (a // state.N, a % state.N)) for a, p in act_probs])
        return act_probs, value

    def fit_one(self, batch: list, lr: float, gamma: float = 1.0) -> tuple[float, float]:
        state_batch, mcts_probs, winner_batch, *next_state_batch = batch
        next_state_given = len(next_state_batch)
        
        state_batch = torch.FloatTensor(np.ascontiguousarray(state_batch)).to(self.device)
        mcts_probs = torch.FloatTensor(np.ascontiguousarray(mcts_probs)).to(self.device)
        winner_batch = torch.FloatTensor(np.ascontiguousarray(winner_batch)).to(self.device)

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        log_act_probs, value = self.policy_value_net(state_batch)
        if next_state_given:
            next_state_batch = next_state_batch[0]
            next_state_batch = torch.FloatTensor(np.ascontiguousarray(next_state_batch)).to(self.device)
            _, next_value = self.policy_value_net(next_state_batch)
            winner_batch += gamma * next_value.view(-1).detach()

        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = policy_loss_fn(mcts_probs, log_act_probs)
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        entropy = entropy_fn(log_act_probs)
        return loss.item(), entropy.item()

    def get_policy_param(self) -> dict:
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file: str):
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)
