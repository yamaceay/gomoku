from .gomoku import Gomoku
from .mcts import Tree, Node, uct_score
import numpy as np
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError
class Player:
    def next_move(self, _: Gomoku) -> tuple[int, int]:
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

    def next_move_probs(self, game: Gomoku):
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
        
        get_reward = lambda child: uct_score(self.tree.root, child, C=0)
        get_action = lambda child: child.state.get_history()[-1]
        get_reward_action = lambda child: (get_reward(child), get_action(child))
        rewards_actions = map(get_reward_action, self.tree.root.children)
        return list(reversed(sorted(rewards_actions, key=lambda x: x[0])))
    
    def next_move(self, game: Gomoku):
        move_probs = self.next_move_probs(game)
        if game.player == 1:
            _, action = move_probs[0]
        else:
            _, action = move_probs[-1]
        return action