import numpy as np
import copy
from .gomoku import Gomoku
from typing import Callable

def prior_fn(board: Gomoku):
    actions = board.actions()
    action_probs = np.ones(len(actions))/len(actions)
    return zip(actions, action_probs)

class Node(object):
    def __init__(self, parent, p: float = 1.0, gamma: float = 1.0):
        self.parent = parent
        self.children = {}
        self.n = 0
        self.Q = 0
        self.p = p
        
        self.gamma = gamma

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def select(self, C: float = 5):
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(C))

    def update(self, reward):
        self.n += 1
        self.Q += self.gamma * (reward - self.Q) / self.n

    def update_recursive(self, reward):
        if self.parent:
            self.parent.update_recursive(-reward)
        self.update(reward)

    def get_value(self, C: float = 5):
        exploitation_term = self.Q
        exploration_term = self.p * np.sqrt(self.parent.n) / (1 + self.n)
        return exploitation_term + C * exploration_term

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None


class Tree(object):
    def __init__(self, prior_fn: Callable, iterations: int = 10000, policy_kwargs: dict = {}, depth: int = 1000):
        self.root = Node(None, 1.0)
        self.prior_fn = prior_fn
        self.policy_kwargs = policy_kwargs
        self.iterations = iterations
        self.depth = depth

    def _playout(self, state: Gomoku):
        node = self.root
        while not node.is_leaf():
            action, node = node.select(**self.policy_kwargs)
            state.play(action)

        action_probs = self.prior_fn(state)
        if not state.fin():
            node.expand(action_probs)
        reward = self._evaluate_rollout(state)
        node.update_recursive(-reward)

    def _evaluate_rollout(self, state: Gomoku):
        player = state.player
        for i in range(self.depth):
            if state.fin():
                break
            actions = state.actions()
            action = actions[0]
            state.play(action)
        else:
            print("WARNING: rollout reached move limit")
        return state.score() * player

    def get_move(self, state: Gomoku):
        for n in range(self.iterations):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self.root.children.items(),
                   key=lambda act_node: act_node[1].n)[0]

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)


class UCT_Player(object):
    def __init__(self, prior_fn: Callable, policy_kwargs: dict = {}, iterations: int = 2000, depth: int = 1000):
        self.mcts = Tree(
            prior_fn=prior_fn, 
            policy_kwargs=policy_kwargs, 
            iterations=iterations,
            depth=depth,
        )

    def next_move(self, board: Gomoku):
        if not board.fin():
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

if __name__ == '__main__':    
    game_kwargs = {
        "M": 8,
        "N": 8,
        "K": 5,
    }
    
    game = Gomoku(**game_kwargs)
    game.set_play_only()
    
    uct = UCT_Player(
        prior_fn=prior_fn,
        policy_kwargs={'C': 5}, 
        iterations=5000,
    )
    
    while not game.fin():
        action = uct.next_move(game)
        game.play(action)
        print(game)
        # uct.set_player_ind(game.current_player)
        # end, winner = game.game_end()
        # if end:
        #     print(winner)
        #     break
        # print(uct)
        # game.move(action)
        # game_copy.play(tuple(game.move_to_location(action)))
        # print(game_copy)
    
    # game = Gomoku(**game_kwargs)
    # while not game.fin():
    #     action = uct.next_move(game)
    #     game.play(action)
    #     print(game)