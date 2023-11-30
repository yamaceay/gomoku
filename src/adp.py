import os
import numpy as np
from .mcts import sortfn
from .players import Player
import torch
from .gomoku import Gomoku
from .patterns import PB_DICT, WIN_ENCODE, revp
from .zero import AlphaZeroPlayer
import random
import logging

HIDDEN_DIM = 100
INPUT_DIM = 2 * (5 * (len(PB_DICT) - 1) + 1) + 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv_Net(torch.nn.Module):
    def __init__(self, conv_params, ff_params, **kwargs):
        super(Conv_Net, self).__init__()
        
        self.layer = kwargs.get('layer', torch.nn.Linear)
        self.activation_fn = kwargs.get('activation_fn', torch.nn.Tanh)
        self.loss_fn = kwargs.get('loss_fn', 'mse')
        if self.loss_fn == 'mse':
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        elif self.loss_fn == 'mae':
            self.loss_fn = torch.nn.L1Loss(reduction='sum')
            
        self.layers = []
        # for i in range(len(conv_params) - 1):
        #     self.layers += [
        #         torch.nn.Conv2d(conv_params[i], conv_params[i+1], kernel_size=3, padding=1),
        #         torch.nn.MaxPool2d(kernel_size=2, stride=2),
        #         torch.nn.Flatten(),
        #         self.activation_fn(),
        #     ]
        
        for i in range(len(ff_params) - 1):
            self.layers += [
                self.layer(ff_params[i], ff_params[i+1]),
                self.activation_fn(),
            ]
        
        self.model = torch.nn.Sequential(*self.layers)
        self.model = self.model.to(device)

        self.lr = kwargs.get('lr', 0.1)
        self.optimizer_fn = kwargs.get('optimizer_fn', torch.optim.Adam)
        self.optimizer = self.optimizer_fn(self.model.parameters(), lr=self.lr)
        

    def __call__(self, x):
        return self.model(x)
    
class FF_Net(torch.nn.Module):
    def __init__(self, params, **kwargs):
        super(FF_Net, self).__init__()
        
        self.layer = kwargs.get('layer', torch.nn.Linear)
        self.activation_fn = kwargs.get('activation_fn', torch.nn.Tanh)
        self.loss_fn = kwargs.get('loss_fn', 'mse')
        if self.loss_fn == 'mse':
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        elif self.loss_fn == 'mae':
            self.loss_fn = torch.nn.L1Loss(reduction='sum')
            
        self.layers = []
        for i in range(len(params) - 1):
            self.layers += [
                self.layer(params[i], params[i+1]),
                self.activation_fn(),
            ]
        
        self.model = torch.nn.Sequential(*self.layers)
        self.model = self.model.to(device)

        self.lr = kwargs.get('lr', 0.1)
        self.optimizer_fn = kwargs.get('optimizer_fn', torch.optim.Adam)
        self.optimizer = self.optimizer_fn(self.model.parameters(), lr=self.lr)
        

    def __call__(self, x):
        return self.model(x)

    # def train(self, x, y):
    #     x = x.to(device)
    #     y = y.to(device)
    #     y_pred = self.forward(x)
    #     loss = self.loss_fn(y_pred, y)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.item()
    
    # def evaluate(self, x, y):
    #     y_pred = self.forward(x)
    #     loss = self.loss_fn(y_pred, y)
    #     return loss.item()
    
class ADP_Value_Net(FF_Net):
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        if not self.model_path:
            raise Exception('model_path is required')
        
        self.alpha = kwargs.pop('alpha', 1)
        self.gamma = kwargs.pop('gamma', 0.9)
        self.magnify = kwargs.pop('magnify', 1)
        self.n_steps = kwargs.pop('n_steps', 1)
        self.logger = kwargs.pop('logger', logging.getLogger(__name__))
        
        super(ADP_Value_Net, self).__init__(
            params=[INPUT_DIM, HIDDEN_DIM, 1], 
            **kwargs
        )
        
        self.model = self.model.to(device)
        try:
            self.load_model()
        except Exception as e:
            self.logger.error(e)
            self.logger.info('Initializing new model')
    
    def __call__(self, state: Gomoku):
        if state.fin():
            reward = state.score()
            return torch.FloatTensor([reward]).to(device)
        
        value_list, end_result = self.extract_values(state)
        if end_result is not None:
            return torch.FloatTensor([end_result]).to(device)
        features = self.extract_features(state, value_list)
        return self.model(features)
    
    def opt(self, *states: list[Gomoku]):
        assert len(states) > 1, "At least 2 states are required"
        
        V_func = lambda i: self(states[i]).to(device) * (self.gamma ** i)
        [V, *V_next] = list(map(V_func, range(len(states))))

        loss = self.alpha * (sum(V_next) - V)
                
        loss_squared = loss ** 2
        return loss_squared
    
    def extract_values(self, state: Gomoku):
        assert len(state.line_cache), "Line cache is empty"
        value_list = {len(pattern): [] for pattern in PB_DICT}
        for length in state.get_line_cache():
            for position in state.get_line_cache(length):
                for direction in state.get_line_cache(length, position):
                    _, values = state.get_line_cache(length, position, direction)
                    if values in WIN_ENCODE:
                        return {}, -1
                    if values in map(revp, WIN_ENCODE):
                        return {}, 1
                    value_list[length] += [values]
        return value_list, None  
    
    def extract_features(self, state: Gomoku, value_list: dict):        
        counts = []
        for pattern_o in PB_DICT:
            values = value_list[len(pattern_o)]
            pattern_x = revp(pattern_o)
            first_count, second_count = 0, 0
            for value in values:
                len_diff = len(pattern_x) - len(value)
                assert 0 <= len_diff <= 1, "Length difference of pattern vs. line: {}".format(len_diff)
                add_bound = len_diff > 0
                if value == pattern_x[:len(value)]:
                    if not add_bound or pattern_x[len(value)] == 'o':
                        first_count += 1
                if value == pattern_o[:len(value)]:
                    if not add_bound or pattern_o[len(value)] == 'x':
                        second_count += 1
            counts += [first_count, second_count]
        
        to_bit = lambda c: int(c > 0)
        to_bits = lambda c: [int(c > i) for i in range(4)] + [int((c-4)/2) if c>4 else 0]
        [vcfo, vcfx, *rest] = counts
        features = [to_bit(vcfo), to_bit(vcfx)]
        for c in rest:
            features += to_bits(c)
        features += [int(state.player == 1), int(state.player == -1)]
        return torch.FloatTensor(features).to(device)
    
    def get_rewards_actions(self, state: Gomoku) -> list[tuple[float, tuple[int, int]]]:  
        actions = state.actions(only_adjacents=True)
        assert len(actions), "No moves available"
        
        rewards_actions = []
        for action in actions:
            state_new = state.copy()
            state_new.play(action)
            value = self(state_new)
            if value.device != device:
                value = value.to(device)
            rewards_actions.append((value, action))
            
        rewards_actions = sortfn(rewards_actions, lambda ra: ra[0])
        return rewards_actions
    
    def train_body(self, history: list[Gomoku]):
        losses = []
        last_step = len(history)-1
        for i in range(last_step-1, -1, -1):
            states = history[i:]
            if i + self.n_steps < last_step:
                states = states[:self.n_steps+1]
            loss = self.opt(*states)
            losses += [loss]
            
        losses = torch.stack(losses).to(device)
        objective = torch.zeros_like(losses).to(device)
        loss = self.loss_fn(losses, objective)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().item()
    
    def train_by_zero(self, state: Gomoku, policy_network):
        zero_player = AlphaZeroPlayer(
            M = state.M, 
            N = state.N, 
            K = state.K
        )
        
        history = [state.copy()]
        if np.random.random() < 0.5:
            action = zero_player.next_move(state)
            state.play(action)
            history += [state.copy()]
            
        while not state.fin():
            action = policy_network(state, self)
            state.play(action)                
            history += [state.copy()]
            if state.fin():
                break
            action = zero_player.next_move(state)
            state.play(action)
            history += [state.copy()]
        
        return self.train_body(history)
    
    def train(self, state: Gomoku, policy_network):
        history = [state.copy()]
        while not state.fin():
            action = policy_network(state, self)
            state.play(action)
            history += [state.copy()]
            
        return self.train_body(history)
    
    def load_model(self, filepath: str = None):
        if filepath is None:
            filepath = self.model_path
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath))
            self.logger.info('Loaded existing model at '+filepath)
        else:
            self.logger.info('File '+filepath+' does not exist')
        
    def save_model(self, filepath: str = None):
        if filepath is None:
            filepath = self.model_path
        torch.save(self.model.state_dict(), filepath)
        self.logger.info('Saved model at '+filepath)
    
class ADP_Policy:
    def __init__(self, **kwargs):
        self.epsilon = kwargs.get('epsilon', 0.)
    
    def __call__(self, state: Gomoku, value_network: ADP_Value_Net) -> list[tuple[float, tuple[int, int]]]:            
        rewards_actions = value_network.get_rewards_actions(state)
        if random.random() < self.epsilon:
            probs = np.random.random(len(rewards_actions))
            probs /= probs.sum()
            rewards_actions = [(probs[i], rewards_actions[i][1]) for i in range(len(rewards_actions))]
            rewards_actions = sortfn(rewards_actions, lambda x: x[0])
        return rewards_actions
  
class ADP_Player(Player):
    def __init__(self, model_path: str, value_network_kwargs, policy_network_kwargs): 
        self.value_network = ADP_Value_Net(model_path=model_path, **value_network_kwargs)
        self.policy_network = ADP_Policy(**policy_network_kwargs)
    
    def next_move_probs(self, game: Gomoku) -> tuple[int, int]:
        return self.policy_network(game, self.value_network)
  
# class UCT_ADP_Player(UCT_Player):
#     def __init__(self, max_depth=10, model=None, **kwargs):
#         super(UCT_ADP_Player, self).__init__(**kwargs)
    
#         self.max_depth = max_depth
#         self.model: ADP_Player = model
    
#     def simulate(self, node: Node) -> float:
#         state = node.state.copy()
#         for _ in range(self.max_depth):
#             if state.fin():
#                 break
            
#             action = self.model.next_move(state)
#             state.play(action)
            
#         if state.fin():
#             return state.score()
        
#         value = self.model.\
#             value_network.\
#             forward(state).\
#             cpu().\
#             detach().\
#             item()

#         return value
    
#     def next_move(self, game: Gomoku):
#         self.tree = Tree(game, **self.tree_kwargs)
        
#         signal.signal(signal.SIGALRM, timeout_handler)
#         signal.alarm(self.timeout_ms // 1000)  # alarm is set with seconds
        
#         try:
#             for _ in range(self.iterations):
#                 node = self.tree.select(policy=self.policy, policy_kwargs=self.policy_kwargs)
#                 value = self.simulate(node)
#                 self.tree.backpropagate(node, value)
#         except TimeoutError:
#             pass
        
#         best_child = max(self.tree.root.children, key=lambda child: child.Q)
#         return best_child.state.history[-1]    