import numpy as np
from .gomoku import Gomoku
from typing import Callable
from .adp import ADP_Player
from .players import Player
from .patterns import sortfn, Pattern

def uniform_prior(state: Gomoku):
    actions = state.actions()
    probs = np.ones(len(actions))/len(actions)
    return zip(probs, actions)

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
                 prior_fn: Callable, 
                 iterations: int = 10000, 
                 policy_kwargs: dict = {}, 
                 max_depth: int = 1000,
                 gamma: float = 1.0,
                 value_fn: ADP_Player = None,
                 play_random: bool = True,
                 play_epsilon: float = 0.0,
                 ):
        
        self.root = Node()
        self.prior_fn = prior_fn
        self.policy_kwargs = policy_kwargs
        self.iterations = iterations
        self.gamma = gamma
        
        self.value_fn = value_fn
        self.max_depth = max_depth
        self.play_random = play_random
        self.play_epsilon = play_epsilon
        
        if not self.play_random:
            assert self.value_fn is not None, "Must provide value_fn if not playing random"

    def iterate(self, state: Gomoku):
        node = self.root
        while not node.is_terminal():
            action, node = node.select(**self.policy_kwargs)
            state.play(action)

        probs_actions = self.prior_fn(state)
        if not state.fin():
            node.expand(probs_actions)
        reward = self.rollout(state)
        self.backpropagate(node, -reward)

    def rollout(self, state: Gomoku):
        player = state.player
        for _ in range(self.max_depth):
            if state.fin():
                break
            action = state.actions()[0]
            if not self.play_random:
                action = self.value_fn.next_move(state, self.play_epsilon)
            state.play(action)
        
        score = state.score()
        if self.value_fn is not None:
            score = self.value_fn(state)
            score = score.cpu().detach().item()
        score *= player
        return score
    
    def backpropagate(self, node: Node, reward: float):
        while node is not None:
            node.n += 1
            node.Q += self.gamma * (reward - node.Q) / node.n
            reward = -reward
            node = node.parent

    def get_move_probs(self, state: Gomoku):
        for _ in range(self.iterations):
            self.iterate(state.copy())
        actions, probs = zip(*[(act, node.n) for act, node in self.root.children.items()])
        probs = np.exp(probs - np.max(probs))
        probs /= np.sum(probs)
        return sortfn(zip(probs, actions))

class UCT_Player(Player):
    def __init__(self, 
                 prior_fn: Callable = uniform_prior, 
                 policy_kwargs: dict = {}, 
                 iterations: int = 2000, 
                 max_depth: int = 1000,
                 value_fn: ADP_Player = None,
                 play_random: bool = True,
                 play_epsilon: float = 0.0,
                 ):
        self.tree = Tree(
            value_fn=value_fn,
            prior_fn=prior_fn, 
            policy_kwargs=policy_kwargs, 
            iterations=iterations,
            max_depth=max_depth,
            play_random=play_random,
            play_epsilon=play_epsilon,
        )
        self.history = []

    def update_history(self, state: Gomoku) -> bool:
        prev_history = self.history
        self.history = state.history()
        if len(self.history) > len(prev_history):
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
        return True

    def next_move_probs(self, state: Gomoku):
        self.update_history(state)
        move_probs = self.tree.get_move_probs(state)
        return move_probs

if __name__ == '__main__':    
    from .adp import ADP_Conv_Player
    game_kwargs = {
        "M": 8,
        "N": 8,
        "K": 5,
    }
    
    game = Gomoku(**game_kwargs)
    game.set_play_only()
    
    adp = ADP_Conv_Player(game_kwargs=game_kwargs)
    
    uct = UCT_Player(
        iterations=5000,
        policy_kwargs={'C': 5},
        value_fn=adp,
    )
    
    while not game.fin():
        action = uct.next_move(game)
        game.play(action)
        print(game)