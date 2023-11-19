import os
import torch
from .gomoku import Gomoku
from .patterns import PB_DICT, lti
import random

DIR_PATH = "./models"

HIDDEN_DIM = 100
INPUT_DIM = 2 * (5 * (len(PB_DICT) - 1) + 1) + 2

class ShallowNN(torch.nn.Module):
    def __init__(self, params, **kwargs):
        super(ShallowNN, self).__init__()
        
        self.layer = kwargs.get('layer', torch.nn.Linear)
        self.activation_fn = kwargs.get('activation_fn', torch.nn.Sigmoid)
        self.loss_fn = kwargs.get('loss_fn', 'mse')
        if self.loss_fn == 'mse':
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        
        self.optimizer_fn = kwargs.get('optimizer_fn', torch.optim.SGD)
        self.lr = kwargs.get('lr', 0.1)
        
        self.layers = []
        for i in range(len(params) - 1):
            self.layers += [
                self.layer(params[i], params[i+1]),
                self.activation_fn(),
            ]
        
        self.model = torch.nn.Sequential(*self.layers)
        self.optimizer = self.optimizer_fn(self.model.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)

    def train(self, x, y):
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, x, y):
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return loss.item()
    
class ValueNetwork(ShallowNN):
    def __init__(self, **kwargs):
        self.alpha = kwargs.pop('alpha', 1)
        self.magnify = kwargs.pop('magnify', 1)
        self.model_path = kwargs.pop('model_path', os.path.join(DIR_PATH, "best.h5"))
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
        
        super(ValueNetwork, self).__init__(
            params=[INPUT_DIM, HIDDEN_DIM, 1], 
            **kwargs
        )
    
    def forward(self, state: Gomoku):
        if state.fin():
            reward = abs(state.score())
            return torch.FloatTensor([reward])
        
        features = self.extract_features(state)
        return self.model(features)
    
    def train(self, prev_state: Gomoku, state: Gomoku, reward: float):
        prev_V = self.forward(prev_state)
        V = self.forward(state)
        lr = self.lr
        if V == 0:
            lr *= self.magnify
        optimizer = self.optimizer_fn(self.model.parameters(), lr=lr)
        loss = self.alpha * (reward + V - prev_V)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def extract_features(self, state: Gomoku):
        all_indices = [(x, y) 
            for x in range(state.M)
            for y in range(state.N)
        ]
        
        pattern_lengths = set([len(pattern) for pattern in PB_DICT])
        value_list = {length: [] for length in pattern_lengths}
        for dx, dy in state.directions:
            for x, y in all_indices:
                for length in pattern_lengths:
                    values = state.get_line((x, y), (dx, dy), length)     
                    if len(values):
                        value_list[length] += [values]
        
        counts = []
        for pattern in PB_DICT:
            values = value_list[len(pattern)]
            pattern_first = list(map(lti, pattern))
            pattern_second = [-c for c in pattern_first]
            first_count, second_count = 0, 0
            for value in values:
                if all([v == p for v, p in zip(value, pattern_first)]):
                    first_count += 1
                if all([v == p for v, p in zip(value, pattern_second)]):
                    second_count += 1
            counts += [first_count, second_count]
        
        to_bit = lambda c: int(c > 0)
        to_bits = lambda c: [int(c > i) for i in range(4)] + [int((c-4)/2) if c>4 else 0]
        [vcfo, vcfx, *rest] = counts
        features = [to_bit(vcfo), to_bit(vcfx)]
        for c in rest:
            features += to_bits(c)
        features += [int(state.player == 1), int(state.player == -1)]
        return torch.FloatTensor(features)
    
    def load_model(self, filepath: str):
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath))
            print('Loaded existing model at '+filepath)
        else:
            print('File '+filepath+' does not exist')
        
    def save_model(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)
        print('Saved model at '+filepath)
    
class PolicyNetwork:
    def __init__(self, **kwargs):
        self.epsilon = kwargs.get('epsilon', 0.)
    
    def forward(self, state: Gomoku, value_network: ValueNetwork) -> tuple[int, int]:            
        rewards_actions = get_rewards_actions(state, value_network)
        _, best_action = rewards_actions[0]
        if random.random() < self.epsilon:
            _, best_action = random.choice(rewards_actions)
        return best_action
        
def get_rewards_actions(state: Gomoku, value_network: ValueNetwork) -> list[tuple[float, tuple[int, int]]]:  
    actions = state.actions(only_adjacents=True)
    if not len(actions):
        actions = state.actions()
    
    rewards_actions = []
    for action in actions:
        state_new = state.copy()
        state_new.play(action)
        value = value_network.forward(state_new)
        rewards_actions.append((value, action))
        
    rewards_actions = list(reversed(sorted(rewards_actions)))
    return rewards_actions
      
def train_one_episode(state: Gomoku, value_network: ValueNetwork, policy_network: PolicyNetwork):
    losses = []
    while not state.fin():
        prev_state = state.copy()
        action = policy_network.forward(state, value_network)
        state.play(action)
        reward = int(state.score() > 0)
        loss = value_network.train(prev_state, state, reward)
        losses += [loss]
    return losses

def comp_models_in_many_games(game_kwargs, last_model: ValueNetwork, best_model: ValueNetwork, policy: PolicyNetwork, n_games: int = 100):
    stats = []
    for _ in range(n_games):
        last_model_starts = random.random() < .5
        if last_model_starts:
            model1 = last_model
            model2 = best_model
        else:
            model1 = best_model
            model2 = last_model
        game = Gomoku(**game_kwargs)
        while not game.fin():
            action = policy.forward(game, model1)
            game.play(action)
            if game.fin():
                break
            action = policy.forward(game, model2)
            game.play(action)
        win = int(game.score() > 0)
        stats += [(win, last_model_starts)]
    return stats
      
def train_adp(game_kwargs, value_network_kwargs, policy_network_kwargs, eval: bool = True, epochs: int = 2000, checkpoint: int = 100, n_test_games: int = 5):
    best_model_path = os.path.join(DIR_PATH, "best.h5")
    current_model = ValueNetwork(**value_network_kwargs)
    if os.path.exists(best_model_path):
        current_model.load_model(best_model_path)
    elif eval:
        raise Exception("No best model found")
    
    policy = PolicyNetwork(**policy_network_kwargs)
    
    for i in range(1, epochs+1):
        game = Gomoku(**game_kwargs)
        losses = train_one_episode(game, current_model, policy)
        print("Epoch {}, Loss {}".format(i, sum(losses)))
        
        if i % checkpoint == 0:
            new_path = os.path.join(DIR_PATH, "epoch_%s.h5" % i)
            current_model.save_model(new_path)
            
            if eval:
                best_model = ValueNetwork(**value_network_kwargs)
                best_model.load_model(best_model_path)
                
                stats = comp_models_in_many_games(game_kwargs, current_model, best_model, policy, n_games=n_test_games)
                sum_win = sum([win for win, _ in stats])
                if sum_win > n_test_games / 2:
                    print("Current model won {} of {}".format(sum_win, n_test_games))
                    
                    current_model.save_model(best_model_path)      

if __name__ == "__main__":
    train_adp(epochs=1, eval=False, game_kwargs={
        'M': 10,
        'N': 10,
        'K': 5,
        'ADJ': 2,
    }, value_network_kwargs={
        'alpha': 0.9,
        'magnify': 5,
    }, policy_network_kwargs={
        'epsilon': 0.1,
    })

    