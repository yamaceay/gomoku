from .gomoku import Gomoku
from .mcts import Tree, Node, uct_score, sortfn
import numpy as np
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError
class Player:
    def next_move_probs(self, _: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        raise NotImplementedError
    def next_move(self, game: Gomoku) -> tuple[int, int]:
        rewards_actions = self.next_move_probs(game)
        if game.player == 1:
            _, best_action = rewards_actions[0]
        else:
            _, best_action = rewards_actions[-1]
        return best_action
class RandomPlayer(Player):
    def next_move_probs(self, game: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        moves = game.actions()
        probs = np.random.random(len(moves))
        probs /= probs.sum()
        rewards_actions = [(probs[i], moves[i]) for i in range(len(moves))]
        rewards_actions = sortfn(rewards_actions, lambda x: x[0])
        return rewards_actions
            
class UCT_Player(Player):
    def __init__(self, iterations=10000, timeout_ms=0, policy=uct_score, policy_kwargs={}, tree_kwargs={}):
        self.iterations = iterations
        self.policy = policy
        self.policy_kwargs = policy_kwargs
        self.tree_kwargs = tree_kwargs
        self.timeout_ms = timeout_ms

    def simulate(self, node: Node) -> float:
        return self.tree.simulate(node)

    def next_move_probs(self, game: Gomoku):
        self.tree = Tree(game, **self.tree_kwargs)
        first_player = self.tree.root.state.player
        
        if self.timeout_ms > 0:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_ms // 1000)  # alarm is set with seconds
        
        try:
            for _ in range(self.iterations):
                node = self.tree.select(policy=self.policy, policy_kwargs=self.policy_kwargs)
                if not node.is_fully_expanded() and not node.is_terminal():
                    node = self.tree.expand(node)
                value = self.simulate(node)
                self.tree.backpropagate(node, value)
        except TimeoutError:
            pass
        
        get_reward = lambda child: uct_score(self.tree.root, child, C=0) * first_player
        get_action = lambda child: child.state.get_history()[-1] 
        get_reward_action = lambda child: (get_reward(child), get_action(child))
        rewards_actions = map(get_reward_action, self.tree.root.children)
        return sortfn(rewards_actions, key=lambda x: x[0])