import numpy as np

from .gomoku import Gomoku, sortfn
from typing import Callable
from .player import Player
from .calc import dirichlet_noise, softmax
from operator import itemgetter

def uniform_probs(state: Gomoku):
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
    
    def __repr__(self) -> str:
        return f"Node(Parent={self.parent}, # Children={len(self.children)}), N={self.n}, Q={self.Q}, P={self.p}"

class Tree(object):
    def __init__(self, 
                 iterations: int, 
                 policy_value_fn: Callable = None,
                 k_ucb: float = 5,
                 gamma: float = 1.0,
                 ):
        
        self.root = Node()
        self.k_ucb = k_ucb
        self.iterations = iterations
        self.gamma = gamma
        
        if policy_value_fn is None:
            self.policy_value_fn = lambda s: (uniform_probs(s), self.rollout(s))
        else:
            self.policy_value_fn = policy_value_fn

    def iterate(self, state: Gomoku):
        node = self.root
        while not node.is_terminal():
            action, node = node.select(self.k_ucb)
            state.play(action)
        
        player = state.player
        
        if state.fin():
            reward = state.score() * player
        
        else:
            probs_actions, reward = self.policy_value_fn(state)
            node.expand(probs_actions)
            reward *= player
        
        self.backpropagate(node, -reward)
    
    def rollout(self, state: Gomoku) -> float:
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

class Deep_Player(Player):
    def __init__(self, 
                 iterations: int, 
                 policy_value_fn: Callable = None, 
                 k_ucb: float = 5, 
                 temp: float = .001,
                 memory: bool = False,
                 ):
        
        self.noise = dirichlet_noise
        self.tree = Tree(
            iterations=iterations,
            policy_value_fn=policy_value_fn,
            k_ucb=k_ucb,
        )
        
        self.temp = temp
        self.history = []
        self.memory = memory

    def reuse_tree(self, state: Gomoku) -> bool:
        prev_history = list(self.history)
        self.history = list(state.history)
        if self.memory and len(self.history) > len(prev_history):
            invalid_match = False
            for h1, h2 in zip(prev_history, self.history):
                if h1 != h2:
                    invalid_match = True
                    break
            if not invalid_match:
                rest_history = self.history[len(prev_history):]
                not_found = False
                for move in rest_history:
                    if move not in self.tree.root.children:
                        not_found = True
                        break
                    self.tree.root = self.tree.root.children[move]
                    self.tree.root.parent = None
                if not not_found:
                    return True
        self.tree.root = Node()
        return False

    def next_move_probs(self, state: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        self.reuse_tree(state)
        move_probs = self.tree.get_move_probs(state, temp=self.temp)
        move_probs = sortfn(move_probs)
        return move_probs
    
    def next_move_for_train(self, state: Gomoku, epsilon: float = .0) -> tuple[int, int, list[tuple[float, tuple[int, int]]]]:
        self.reuse_tree(state)
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