from .gomoku import Gomoku
from .mcts import Tree, Node, uct_score
import numpy as np
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError
class Player:
    def next_move(self, game: Gomoku) -> tuple[int, int]:
        raise NotImplementedError

class RandomPlayer(Player):
    def next_move(self, game: Gomoku) -> tuple[int, int]:
        moves = game.actions()
        random_move  = np.random.randint(0, len(moves))
        return moves[random_move]
            
class UCT_Player(Player):
    def __init__(self, iterations=10000, timeout_ms=5000, policy=uct_score, policy_kwargs={}, tree_kwargs={}):
        self.iterations = iterations
        self.policy = policy
        self.policy_kwargs = policy_kwargs
        self.tree_kwargs = tree_kwargs
        self.timeout_ms = timeout_ms

    def simulate(self, node: Node) -> float:
        return self.tree.simulate(node)

    def next_move(self, game: Gomoku):
        self.tree = Tree(game, **self.tree_kwargs)
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout_ms // 1000)  # alarm is set with seconds
        
        try:
            for _ in range(self.iterations):
                node = self.tree.select(policy=self.policy, policy_kwargs=self.policy_kwargs)
                value = self.simulate(node)
                self.tree.backpropagate(node, value)
        except TimeoutError:
            pass
        
        best_child = max(self.tree.root.children, key=lambda child: child.Q)
        return best_child.state.history[-1]