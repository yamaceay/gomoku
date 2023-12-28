import numpy as np
from .gomoku import Gomoku
from typing import Callable
from .players import Player
from .patterns import sortfn
from operator import itemgetter

def uniform_probs(state: Gomoku):
    actions = state.actions()
    probs = np.ones(len(actions))/len(actions)
    return zip(probs, actions)

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def dirichlet_noise(x):
    return np.random.dirichlet([.03] * x)

class Node(object):
    def __init__(self, parent = None, p: float = 1.0):
        self.parent = parent
        self.children = {}
        self.n = 0
        self.Q = 0
        self.p = p

    def expand(self, probs_actions: list[tuple[float, tuple[int, int]]]):
        for prob, action in probs_actions:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def select(self, C: float = 5):
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].ucb_score(C))

    def ucb_score(self, C: float = 5):
        exploitation_term = self.Q
        exploration_term = self.p * np.sqrt(self.parent.n) / (1 + self.n)
        return exploitation_term + C * exploration_term

    def is_terminal(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

class Tree(object):
    def __init__(self, 
                 iterations: int, 
                 policy_value_fn: Callable = None,
                 c_puct: float = 5,
                 gamma: float = 1.0,
                 ):
        
        self.root = Node()
        self.c_puct = c_puct
        self.iterations = iterations
        self.gamma = gamma
        
        if policy_value_fn is None:
            self.policy_value_fn = lambda s: (uniform_probs(s), self.rollout(s))
        else:
            self.policy_value_fn = policy_value_fn

    def iterate(self, state: Gomoku):
        node = self.root
        while not node.is_terminal():
            action, node = node.select(self.c_puct)
            state.play(action)
        
        player = state.player
        
        if state.fin():
            reward = state.score() * player
        
        else:
            probs_actions, reward = self.policy_value_fn(state)
            node.expand(probs_actions)
            reward *= player
        
        self.backpropagate(node, -reward)
    
    def rollout(self, state: Gomoku):
        while not state.fin():
            action = state.actions()[0]
            state.play(action)
        return state.score()
    
    def backpropagate(self, node: Node, reward: float):
        while node is not None:
            node.n += 1
            node.Q += self.gamma * (reward - node.Q) / node.n
            reward = -reward
            node = node.parent

    def get_move_probs(self, 
                       state: Gomoku, 
                       temp: float, 
                       ) -> list[tuple[float, tuple[int, int]]]:
        for _ in range(self.iterations):
            self.iterate(state.copy())
        
        actions, probs = zip(*[(act, node.n) for act, node in self.root.children.items()])
        probs = np.array(probs) 
        
        if self.policy_value_fn is not None:
            probs = np.log(probs + 1e-10)
            
        probs = softmax(probs / temp)
        
        return zip(probs, actions)

class UCT_Player(Player):
    def __init__(self, 
                 iterations: int, 
                 policy_value_fn: Callable = None, 
                 c_puct: float = 5, 
                 temp: float = .001,
                 ):
        
        self.noise = dirichlet_noise
        self.tree = Tree(
            iterations=iterations,
            policy_value_fn=policy_value_fn,
            c_puct=c_puct,
        )
        
        self.temp = temp
        self.history = []

    def update_history(self, state: Gomoku) -> bool:
        prev_history = self.history
        self.history = state.get_history()
        if len(self.history) >= len(prev_history):
            for h1, h2 in zip(prev_history, self.history):
                if h1 != h2:
                    self.tree.root = Node()
                    return False
            rest_history = self.history[len(prev_history):]
            for move in rest_history:
                if move not in self.tree.root.children:
                    self.tree.root = Node()
                    return False
                self.tree.root = self.tree.root.children[move]
                self.tree.root.parent = None
        else:
            self.tree.root = Node()
            return False
        return True

    def next_move_probs(self, state: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        self.update_history(state)
        move_probs = self.tree.get_move_probs(state, temp=self.temp)
        move_probs = sortfn(move_probs)
        return move_probs
    
    def next_move(self, state: Gomoku, epsilon: float = .0, get_probs: bool = False) -> tuple[int, int] | list[tuple[float, tuple[int, int]]]:
        probs_actions = self.next_move_probs(state)
        probs, actions = zip(*probs_actions)
        if epsilon != .0:
            probs += epsilon * (self.noise(len(probs)) - probs)
        action_i = np.random.choice(list(range(len(actions))), p=probs)
        action = actions[action_i]
        if get_probs:
            return action, zip(probs, actions)
        return action
    
    def next_move_data(self, state: Gomoku, epsilon: float = .0) -> tuple[int, int, list[tuple[float, tuple[int, int]]]]:
        self.update_history(state)
        probs_actions = self.tree.get_move_probs(state, temp=self.temp)
        probs, actions = zip(*sorted(probs_actions, key=itemgetter(1)))
        action_indices = [a[0] * state.N + a[1] for a in actions]
        probs_dict = np.zeros(state.M * state.N)
        probs_dict[action_indices] = np.array(probs)
        if epsilon != .0:
            probs += epsilon * (self.noise(len(probs)) - probs)
        action_i = np.random.choice(list(range(len(actions))), p=probs)
        action = actions[action_i]
        
        return action, probs_dict
    
if __name__ == "__main__":
    from .policy_value_net import PolicyValueNet
    from .gomoku import Gomoku
    
    reset_probs = False
    curr_starts = False
    
    game_kwargs = {
        'M': 6,
        'N': 6,
        'K': 4,
    }
    
    game = Gomoku(**game_kwargs)
    
    net = PolicyValueNet(
        game_kwargs['M'], game_kwargs['N'],
        model_file='_zero/models/current_policy.model',
    )
    
    pure_player = UCT_Player(
        c_puct = 5,
        iterations = 5000,
        temp = .001,
    )
    
    policy_value_fn = net.policy_value_fn_sorted
    if reset_probs:
        policy_value_fn = lambda s: (uniform_probs(s), net.policy_value_fn_sorted(s)[1])
    
    curr_player = UCT_Player(
        policy_value_fn = policy_value_fn, 
        c_puct = 5,
        iterations = 400,
        temp = .001,
    )
    
    if not curr_starts:
        action = pure_player.next_move(game)
        game.play(action)
        print(game)
        
    while not game.fin():
        action = curr_player.next_move(game)
        game.play(action)
        print(game)
        if game.fin():
            break
        action = pure_player.next_move(game)
        game.play(action)
        print(game)
    