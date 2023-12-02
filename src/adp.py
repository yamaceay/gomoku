import os
import numpy as np
from .mcts import sortfn
from .players import Player
import torch
from .gomoku import Gomoku
from .patterns import PB_DICT, WIN_ENCODE, revp
from .zero import AlphaZeroPlayer
import logging

HIDDEN_DIM = 100
INPUT_DIM = 2 * (5 * (len(PB_DICT) - 1) + 1) + 2
OUTPUT_DIM = 1

INPUT_CHANNELS = 4
HIDDEN_CHANNELS = 32
OUTPUT_CHANNELS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Net(torch.nn.Module):
    def __init__(self, **kwargs):
        self.model_path = kwargs.pop('model_path', None)
        self.logger = kwargs.pop('logger', logging.getLogger(__name__))
        self.lr = kwargs.pop('lr', 0.1)
       
        super(Net, self).__init__(**kwargs)
        
    def compile_model(self, *layers): 
        self.model = torch.nn.Sequential(*layers)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        
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
        
        self.input_dim = kwargs.pop('input_dim', INPUT_DIM)
        self.hidden_dim = kwargs.pop('hidden_dim', HIDDEN_DIM)
        self.output_dim = kwargs.pop('output_dim', OUTPUT_DIM)
        
        self.conv_out_dim = self.output_channels*self.board_size[0]*self.board_size[1]
        
        super(Conv_Net, self).__init__(**kwargs)
        
        self.compile_model(
            torch.nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid(),
            torch.nn.Conv2d(self.hidden_channels, self.output_channels, kernel_size=1),
            torch.nn.Sigmoid(),
            torch.nn.Flatten(),
            torch.nn.Linear(self.conv_out_dim, self.input_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
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
        
        V_func = lambda i: self(states[i]).to(device) * (self.gamma ** i)
        [V, *V_next] = list(map(V_func, range(len(states))))

        loss = self.alpha * (reward + sum(V_next) - V)
        loss_squared = loss ** 2
        return loss_squared
    
    def train_body(self, history: list[Gomoku]):
        losses = []
        last_state = history.pop()
        reward = last_state.score()
        for i in range(len(history)):
            low, high = i, min(i + self.n_steps, len(history) - 1)
            states = history[low:high+1]
            loss = self.opt(*states, reward=reward)
            losses += [loss]
            
        losses = torch.stack(losses).to(device)
        objective = torch.zeros_like(losses).to(device)
        loss = self.nn.loss_fn(losses, objective)
        
        self.nn.optimizer.zero_grad()
        loss.backward()
        self.nn.optimizer.step()
        
        return loss.cpu().detach().item()
    
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
        
        self.nn = Dense_Net(**kwargs)
        self.nn.model = self.nn.model.to(device)
        
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
    
    def forward(self, state: Gomoku):
        if state.fin():
            reward = state.score()
            return torch.FloatTensor([reward]).to(device)
        
        value_list, end_result = self.extract_values(state)
        if end_result is not None:
            return torch.FloatTensor([end_result]).to(device)
        features = self.extract_features(state, value_list)
        return self.nn(features)
    
class ADP_Conv_Player(ADP_Player):
    def __init__(self, **kwargs):
        super(ADP_Conv_Player, self).__init__()
        
        self.alpha = kwargs.pop('alpha', 1)
        self.gamma = kwargs.pop('gamma', 0.9)
        self.magnify = kwargs.pop('magnify', 1)
        self.n_steps = kwargs.pop('n_steps', 1)
        self.epsilon = kwargs.pop('epsilon', 0.)

        self.nn = Conv_Net(**kwargs)
        self.nn.model = self.nn.model.to(device)
    
    def extract_features(self, state: Gomoku):
        size = (state.M, state.N)
        states = torch.zeros((4, *size))
        states[0] = torch.FloatTensor(state.board == 1)
        states[1] = torch.FloatTensor(state.board == -1)
        history = state.get_history()
        if len(history):
            states[2][history[-1]] = 1
        if state.player == 1:
            states[3] = 1
        states = states.to(device)
        return states.unsqueeze(0)
    
    def forward(self, state: Gomoku):
        if state.fin():
            reward = state.score()
            return torch.FloatTensor([reward]).to(device)
        
        features = self.extract_features(state)
        return self.nn(features).squeeze(0)
    
if __name__ == '__main__':
    M, N, K = 8, 8, 5
    adp_player = ADP_Conv_Player(
        model_path="models_fwd/best.h5",
        alpha=0.9,
        magnify=1,
        gamma=0.9,
        lr=0.001,
        n_steps=1,
        epsilon=0.1,
        M=M,
        N=N,
    )
    
    game = Gomoku(M=M, N=N, K=K)
    
    print(adp_player.nn.state_dict().__dict__)
    while not game.fin():
        action = adp_player.next_move(game)
        game.play(action)
        # print(game, adp_player.next_move_probs(game)[:5])