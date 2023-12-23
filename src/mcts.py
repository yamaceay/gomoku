import numpy as np
from .gomoku import Gomoku
from typing import Callable
from .players import Player
from .patterns import sortfn

def uniform_prior(state: Gomoku):
    actions = state.actions()
    probs = np.ones(len(actions))/len(actions)
    return zip(probs, actions)

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

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
                 prior_fn: Callable = uniform_prior,
                 policy_value_fn: Callable = None,
                 iterations: int = 10000, 
                 policy_kwargs: dict = {}, 
                 gamma: float = 1.0,
                 noise: Callable = lambda x: np.random.dirichlet([.03] * x),
                 ):
        
        self.root = Node()
        self.policy_kwargs = policy_kwargs
        self.iterations = iterations
        self.gamma = gamma
        
        self.prior_fn = prior_fn
        self.policy_value_fn = policy_value_fn
        self.noise = noise

    def iterate(self, state: Gomoku):
        node = self.root
        while not node.is_terminal():
            action, node = node.select(**self.policy_kwargs)
            state.play(action)

        if self.policy_value_fn is not None:
            probs_actions, reward = self.policy_value_fn(state)
            if not state.fin():
                node.expand(probs_actions)
            else:
                reward = state.score()
        else:
            probs_actions = self.prior_fn(state)
            if not state.fin():
                node.expand(probs_actions)
            reward = self.rollout(state)
        self.backpropagate(node, -reward)
    
    def rollout(self, state: Gomoku):
        player = state.player
        while not state.fin():
            action = state.actions()[0]
            state.play(action)
        
        return state.score() * player
    
    def backpropagate(self, node: Node, reward: float):
        while node is not None:
            node.n += 1
            node.Q += self.gamma * (reward - node.Q) / node.n
            reward = -reward
            node = node.parent

    def get_move_probs(self, state: Gomoku, temp: float = .001):
        for _ in range(self.iterations):
            self.iterate(state.copy())
        actions, probs = zip(*[(act, node.n) for act, node in self.root.children.items()])
        if temp != 0:
            probs += temp * (self.noise(len(probs)) - probs)
        probs = softmax(probs)
        return sortfn(zip(probs, actions))

class UCT_Player(Player):
    def __init__(self, 
                 policy_value_fn: Callable = None,
                 prior_fn: Callable = uniform_prior, 
                 policy_kwargs: dict = {}, 
                 iterations: int = 2000, 
                 temp: float = .001,
                 ):
        
        self.noise = lambda x: np.random.dirichlet(0.3*np.ones(x))
        self.tree = Tree(
            policy_value_fn=policy_value_fn,
            prior_fn=prior_fn, 
            policy_kwargs=policy_kwargs, 
            iterations=iterations,
            noise=self.noise,
        )
        
        self.temp = temp
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

    def next_move_probs(self, state: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        self.update_history(state)
        move_probs = self.tree.get_move_probs(state, temp=self.temp)
        return move_probs
    
    def next_move(self, state: Gomoku, epsilon: float = .0) -> tuple[int, int]:
        probs_actions = self.next_move_probs(state)
        probs, actions = zip(*probs_actions)
        if epsilon != .0:
            probs += epsilon * (self.noise(len(probs)) - probs)
        action_i = np.random.choice(list(range(len(actions))), p=probs)
        return actions[action_i]

if __name__ == '__main__':    
    game_kwargs = {
        "M": 8,
        "N": 8,
        "K": 5,
    }
    
    game = Gomoku(**game_kwargs)
    game.set_play_only()
    
    uct = UCT_Player(
        iterations=5000,
        policy_kwargs={'C': 5},
    )
    
    while not game.fin():
        action = uct.next_move(game)
        game.play(action)
        print(game)