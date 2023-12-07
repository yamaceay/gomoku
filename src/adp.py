import os
import numpy as np
from .mcts import sortfn
from .players import Player
import torch
from .gomoku import Gomoku
from .patterns import PB_DICT, revp
from .zero import AlphaZeroPlayer, AlphaZeroConv
import logging

INPUT_DIM = 2 * (5 * (len(PB_DICT) - 1) + 1) + 2 * (2 * len(PB_DICT)) + 2
HIDDEN_DIM = 100
OUTPUT_DIM = 1

PRE_INPUT_DIM = 128
PRE_HIDDEN_DIM = 32
PRE_OUTPUT_DIM = 1

INPUT_CHANNELS = 4
HIDDEN_CHANNELS = 32
OUTPUT_CHANNELS = 64
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(torch.nn.Module):
    def __init__(self, **kwargs):
        self.model_path = kwargs.pop('model_path', None)
        self.logger = kwargs.pop('logger', logging.getLogger(__name__))
        self.lr = kwargs.pop('lr', 0.1)
        self.device = kwargs.pop('device', device)
        
        super(Net, self).__init__(**kwargs)
        
    def compile_model(self, *layers): 
        self.model = torch.nn.Sequential(*layers)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        
        try:
            self.load_model()
        except Exception as e:
            self.logger.error(e)
            self.logger.info('Initializing new model')
    
    def forward(self, x):
        return self.model(x)
    
    def load_model(self, filepath: str = None):
        if filepath is None:
            filepath = self.model_path
        assert filepath is not None, "Filepath is required"
        assert os.path.exists(filepath), f"Filepath does not exist: {filepath}"
        self.model.load_state_dict(torch.load(filepath))
        
    def save_model(self, filepath: str = None):
        if filepath is None:
            filepath = self.model_path
        assert filepath is not None, "Filepath is required"
        torch.save(self.model.state_dict(), filepath)

class Conv_Net(Net):
    def __init__(self, **kwargs):
        self.board_size = (kwargs.pop('M'), kwargs.pop('N'))
        self.input_channels = kwargs.pop('input_channels', INPUT_CHANNELS)
        self.hidden_channels = kwargs.pop('hidden_channels', HIDDEN_CHANNELS)
        self.output_channels = kwargs.pop('output_channels', OUTPUT_CHANNELS)
        
        self.output_dim = kwargs.pop('output_dim', OUTPUT_DIM)
        
        super(Conv_Net, self).__init__(**kwargs)
        
        self.compile_model(
            torch.nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(self.hidden_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(self.hidden_channels, self.output_channels, kernel_size=3),
            torch.nn.BatchNorm2d(self.output_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(self.output_channels, self.output_dim),
            torch.nn.Tanh(),
        )
    
class Dense_Net(Net):
    def __init__(self, **kwargs):
        self.input_dim = kwargs.pop('input_dim', INPUT_DIM)
        self.hidden_dim = kwargs.pop('hidden_dim', HIDDEN_DIM)
        self.output_dim = kwargs.pop('output_dim', OUTPUT_DIM)
        
        super(Dense_Net, self).__init__(**kwargs)
        
        self.compile_model(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
            torch.nn.Tanh(),
        )
    
class ADP_Player(Player):
    def next_move_probs(self, state: Gomoku) -> list[tuple[float, tuple[int, int]]]:  
        actions = state.actions()
        assert len(actions), "No moves available"
        
        rewards_actions = []
        for action in actions:
            state_new = state.copy()
            state_new.play(action)
            value = self(state_new).cpu().detach().item()
            rewards_actions.append((value, action))
        return sortfn(rewards_actions, key=lambda x: x[0])
    
    def __call__(self, state: Gomoku):
        return self.forward(state)
    
    def opt(self, *states: list[Gomoku], **kwargs):
        reward = kwargs.pop('reward', .0)
        
        assert len(states) > 0, "At least 1 state is required"
        
        V_func = lambda i: self(states[i]).to(self.device) * (self.gamma ** i)
        [V, *V_next] = list(map(V_func, range(len(states))))

        sum_V_next = sum(V_next) if len(V_next) else torch.tensor([0.]).to(self.device)

        # print("{} + {:.3f} = {:.3f}".format(reward, sum_V_next.cpu().detach().item(), V.cpu().detach().item()))
        loss = self.alpha * (reward + sum_V_next - V)
        return loss
    
    def train_body(self, history: list[Gomoku]):
        losses = []
        last_state = history.pop()
        reward = last_state.score()
        for i in range(len(history)):
            low, high = i, min(i + self.n_steps, len(history) - 1)
            states = history[low:high+1]
            loss = self.opt(*states, reward=reward)
            losses += [loss]
            
        losses = torch.stack(losses).to(self.device)
        objective = torch.zeros_like(losses).to(self.device)
        loss = self.nn.loss_fn(losses, objective)
        
        self.nn.optimizer.zero_grad()
        loss.backward()
        self.nn.optimizer.step()
        
        return loss.cpu().detach().item(), reward
    
    def train_by_zero(self, state: Gomoku, **kwargs):
        epsilon = kwargs.get('epsilon', .1)
        
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
            action = state.actions()[0]
            if np.random.random() >= epsilon:
                action = self.next_move(state)
            state.play(action)                
            history += [state.copy()]
            if state.fin():
                break
            action = zero_player.next_move(state)
            state.play(action)
            history += [state.copy()]
        
        return self.train_body(history)
    
    def train(self, state: Gomoku, **kwargs):
        epsilon = kwargs.get('epsilon', .1)
        history = [state.copy()]
        while not state.fin():
            action = state.actions()[0]
            if np.random.random() >= epsilon:
                action = self.next_move(state)
            state.play(action)
            history += [state.copy()]
            
        return self.train_body(history)

    
class ADP_Dense_Player(ADP_Player):
    def __init__(self, **kwargs):
        super(ADP_Dense_Player, self).__init__()
        
        self.alpha = kwargs.pop('alpha', 1)
        self.gamma = kwargs.pop('gamma', 0.9)
        self.magnify = kwargs.pop('magnify', 1)
        self.n_steps = kwargs.pop('n_steps', 1)
        self.epsilon = kwargs.pop('epsilon', 0.)
        
        self.device = kwargs.get('device', device)
        
        kwargs.pop('M', None)
        kwargs.pop('N', None)
        kwargs.pop('K', None)
        kwargs.pop('ADJ', None)
        
        self.nn = Dense_Net(**kwargs)
        
    def extract_values(self, state: Gomoku):
        assert len(state.line_cache), "Line cache is empty"
        value_list = {len(pattern): [] for pattern in PB_DICT}
        affected_value_list = {len(pattern): [] for pattern in PB_DICT}
        for length in state.get_line_cache():
            for position in state.get_line_cache(length):
                for direction in state.get_line_cache(length, position):
                    _, values = state.get_line_cache(length, position, direction)
                    if position == state.get_history()[-1]:
                        affected_value_list[length] += [values]
                    # if values in WIN_ENCODE:
                    #     return {}, {}, -1
                    # if values in map(revp, WIN_ENCODE):
                    #     return {}, {}, 1
                    value_list[length] += [values]
        return value_list, affected_value_list, None  
    
    def extract_feature(self, values: list[str], pattern: str) -> tuple[int, int]:
        first_count, second_count = 0, 0
    
        pattern_o, pattern_x = pattern, revp(pattern)
        for value in values:
            len_diff = len(pattern) - len(value)
            assert 0 <= len_diff <= 1, "Length difference of pattern vs. line: {}".format(len_diff)
            add_bound = len_diff > 0
            if value == pattern_x[:len(value)]:
                if not add_bound or pattern_x[len(value)] == 'o':
                    first_count += 1
            if value == pattern_o[:len(value)]:
                if not add_bound or pattern_o[len(value)] == 'x':
                    second_count += 1
        return [first_count, second_count]
    
    def extract_features(self, state: Gomoku, value_list: dict, affected_value_list: dict):        
        counts = []
        not_occurred, occurred = [0, 0], [0, 0]
        if state.player == 1:
            occurred[0] = 1
        else:
            occurred[1] = 1
        occurrences = []
        
        for pattern in PB_DICT:
            values = value_list[len(pattern)]
            counts += self.extract_feature(values, pattern)
            
            affected_values = affected_value_list[len(pattern)]
            affected_new_counts = self.extract_feature(affected_values, pattern)
            
            occurrences += occurred if affected_new_counts[0] > 0 else not_occurred
            occurrences += occurred if affected_new_counts[1] > 0 else not_occurred
        
        to_bit = lambda c: int(c > 0)
        to_bits = lambda c: [int(c > i) for i in range(4)] + [int((c-4)/2) if c>4 else 0]
        [vcfo, vcfx, *rest] = counts
        features = [to_bit(vcfx), to_bit(vcfo)]
        for c in rest:
            features += to_bits(c)
        
        features += occurrences
        features += [int(state.player == 1), int(state.player == -1)]
        return torch.FloatTensor(features).to(self.device)
    
    def forward(self, state: Gomoku):
        if state.fin():
            reward = state.score()
            return torch.FloatTensor([reward]).to(self.device)
        
        value_list, affected_value_list, end_result = self.extract_values(state)
        if end_result is not None:
            return torch.FloatTensor([end_result]).to(self.device)
    
        features = self.extract_features(state, value_list, affected_value_list)
        return self.nn(features)
    
class ADP_Pre_Player(ADP_Player):
    def __init__(self, **kwargs):
        super(ADP_Pre_Player, self).__init__()
        
        self.alpha = kwargs.pop('alpha', 1)
        self.gamma = kwargs.pop('gamma', 0.9)
        self.magnify = kwargs.pop('magnify', 1)
        self.n_steps = kwargs.pop('n_steps', 1)
        self.epsilon = kwargs.pop('epsilon', 0.)

        self.device = kwargs.get('device', device)
        
        self.M = kwargs.pop('M')
        self.N = kwargs.pop('N')
        self.K = kwargs.pop('K')
        kwargs.pop('ADJ', None)
        
        self.conv_nn = AlphaZeroConv(self.M, self.N, self.K)
        
        self.nn = Dense_Net(
            input_dim = PRE_INPUT_DIM,
            hidden_dim = PRE_HIDDEN_DIM,
            output_dim = PRE_OUTPUT_DIM,
        )
        
    def extract_features(self, state: Gomoku):
        features = self.conv_nn.forward(state)
        return torch.FloatTensor(features).to(self.device)
    
    def forward(self, state: Gomoku):
        if state.fin():
            reward = state.score()
            return torch.FloatTensor([reward]).to(self.device)
        
        features = self.extract_features(state)
        return self.nn(features)
class ADP_Conv_Player(ADP_Player):
    def __init__(self, **kwargs):
        super(ADP_Conv_Player, self).__init__()
        
        self.alpha = kwargs.pop('alpha', 1)
        self.gamma = kwargs.pop('gamma', 0.9)
        self.magnify = kwargs.pop('magnify', 1)
        self.n_steps = kwargs.pop('n_steps', 1)
        self.epsilon = kwargs.pop('epsilon', 0.)

        self.device = kwargs.get('device', device)
        
        kwargs.pop('K', None)
        kwargs.pop('ADJ', None)
        
        self.nn = Conv_Net(**kwargs)
    
    def forward(self, state: Gomoku):
        if state.fin():
            reward = state.score()
            return torch.FloatTensor([reward]).to(self.device)
        
        features = state.to_zero_input()
        features = torch.FloatTensor(features).to(self.device)
        return self.nn(features.unsqueeze(0)).squeeze(0)