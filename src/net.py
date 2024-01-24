# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .calc import policy_loss_fn, entropy_fn
from .gomoku import Gomoku, sortfn

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class Policy_Value_Net():
    """policy-value network """
    def __init__(self, 
                 game_kwargs: tuple[int, int, int],
                 model_file: str = None, 
                 device: torch.DeviceObjType = torch.device('cpu')):
        self.device = device
        self.board_width = game_kwargs[0]
        self.board_height = game_kwargs[1]
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        self.policy_value_net = Net(self.board_width, self.board_height).to(device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = np.ascontiguousarray(state_batch)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()

    def policy_value_fn(self, state: Gomoku):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        actions = sorted(state.actions())
        legal_positions = [a[0] * state.N + a[1] for a in actions]
        current_state = np.ascontiguousarray(state.encode().reshape(
                -1, 4, self.board_width, self.board_height))
        log_act_probs, value = self.policy_value_net(
                torch.from_numpy(current_state).to(self.device).float())
        act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value
    
    def policy_value_fn_sorted(self, state):
        act_probs, value = self.policy_value_fn(state)
        act_probs = sortfn([(p, (a // state.N, a % state.N)) for a, p in act_probs])
        return act_probs, value

    def train_step(self, batch, lr, gamma: float = 1.0):
        """perform a training step"""
        # wrap in Variable
        state_batch, mcts_probs, winner_batch, *next_state_batch = batch
        next_state_given = len(next_state_batch)
        
        state_batch = torch.FloatTensor(np.ascontiguousarray(state_batch)).to(self.device)
        mcts_probs = torch.FloatTensor(np.ascontiguousarray(mcts_probs)).to(self.device)
        winner_batch = torch.FloatTensor(np.ascontiguousarray(winner_batch)).to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        if next_state_given:
            next_state_batch = next_state_batch[0]
            next_state_batch = torch.FloatTensor(np.ascontiguousarray(next_state_batch)).to(self.device)
            _, next_value = self.policy_value_net(next_state_batch)
            winner_batch += gamma * next_value.cpu().numpy()
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = policy_loss_fn(mcts_probs, log_act_probs)
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = entropy_fn(log_act_probs)
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file: str):
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
