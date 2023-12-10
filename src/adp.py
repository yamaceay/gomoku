from .mcts import sortfn
from .players import Player
import torch
from .gomoku import Gomoku
from .patterns import PB_DICT, Pattern
from .zero import AlphaZeroConv
from .net import Net, Dense_Net, Conv_Net, Pre_Dense_Net
from tqdm import tqdm
        
class ADP_Player(Player):
    def __init__(self, 
                 nn: Net,
                 game_kwargs: dict[int], 
                 alpha: float = 0.9, 
                 gamma: float = 0.9):
        
        self.nn = nn
        self.device = self.nn.device()
        self.game_kwargs = game_kwargs
        self.alpha = alpha
        self.gamma = gamma
        
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
    
    def __call__(self, state: Gomoku) -> torch.Tensor:
        return self.forward(state)
    
    def train_batch(self, batch: list[str], start: int = 0, disable: bool = True) -> float:
        losses = []
        for game_str in tqdm(batch, 
            desc="Training", 
            leave=False, 
            position=1, 
            disable=disable):
            
            moves = Pattern.loc_to_move(game_str)
            if len(moves) <= start:
                continue
            
            history = []
            game = Gomoku(**self.game_kwargs)
            for move in moves:
                game.play(move)
                state = self(game).to(self.device)
                history += [state]
            history = history[start:]
            
            reward = history[-1].cpu().detach().item()
            for i in range(len(history) - 1):
                [V_curr, V_next] = history[i:i+2]
                if i == len(history) - 2:
                    V_next = torch.tensor([0.]).to(self.device)
                loss = self.alpha * (reward + self.gamma * V_next - V_curr)
                losses += [loss]
        
        losses = torch.stack(losses).to(self.device)
        objective = torch.zeros_like(losses).to(self.device)
        mean_loss = self.nn.loss_fn(losses, objective)
        
        self.nn.optimizer.zero_grad()
        mean_loss.backward()
        self.nn.optimizer.step()
            
        return mean_loss.cpu().detach().item()
    
class ADP_Dense_Player(ADP_Player):
    def __init__(self, 
                 game_kwargs: dict[int],
                 alpha: float = 0.9, 
                 gamma: float = 0.9,
                 **kwargs):
        
        super(ADP_Dense_Player, self).__init__(
            nn=Dense_Net(**kwargs),
            game_kwargs=game_kwargs,
            alpha=alpha,
            gamma=gamma,
        )

        
    def extract_values(self, state: Gomoku):
        assert len(state._line_cache), "Line cache is empty"
        value_list = {len(pattern): [] for pattern in PB_DICT}
        affected_value_list = {len(pattern): [] for pattern in PB_DICT}
        for length in state.get_line_cache():
            for position in state.get_line_cache(length):
                for direction in state.get_line_cache(length, position):
                    values = state.get_line_cache(length, position, direction)
                    if position == state.last_move:
                        affected_value_list[length] += [values]
                    value_list[length] += [values]
        return value_list, affected_value_list, None  
    
    def extract_feature(self, values: list[str], pattern: str) -> tuple[int, int]:
        first_count, second_count = 0, 0
    
        pattern_o, pattern_x = pattern, Pattern.revp(pattern)
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
    def __init__(self, 
                 game_kwargs: dict[int],
                 alpha: float = 1, 
                 gamma: float = 0.9, 
                 **kwargs,
                 ):
        
        self.conv_nn = AlphaZeroConv(
            M = game_kwargs['M'],
            N = game_kwargs['N'],
            K = game_kwargs['K'],
        )
        
        super(ADP_Pre_Player, self).__init__(
            nn=Pre_Dense_Net(**kwargs),
            game_kwargs=game_kwargs,
            alpha=alpha,
            gamma=gamma,
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
    def __init__(self, 
                 game_kwargs: dict[int],
                 alpha: float = 1, 
                 gamma: float = 0.9, 
                 **kwargs):
        
        super(ADP_Conv_Player, self).__init__(
            nn=Conv_Net(
                M = game_kwargs['M'],
                N = game_kwargs['N'], 
                **kwargs
            ),
            game_kwargs=game_kwargs,
            alpha=alpha,
            gamma=gamma,
            **kwargs,
        )
    
    def forward(self, state: Gomoku):
        if state.fin():
            reward = state.score()
            return torch.FloatTensor([reward]).to(self.device)
        
        features = torch.FloatTensor(state.to_zero_input()).to(self.device)
        return self.nn(features.unsqueeze(0)).squeeze(0)