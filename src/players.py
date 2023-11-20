from .gomoku import Gomoku
from .mcts import Tree, Node, uct_score
from .adp import PolicyNetwork, ValueNetwork
import numpy as np
import threading

class TimeoutError(Exception):
    pass

def timeout(seconds):
    def decorator(function):
        def wrapper(*args, **kwargs):
            stop_thread = threading.Event()
            result = None
            def function_with_stop_check(*args, **kwargs):
                nonlocal result
                while not stop_thread.is_set():
                    result = function(*args, **kwargs)
            thread = threading.Thread(target=function_with_stop_check, args=args, kwargs=kwargs)
            thread.start()
            thread.join(timeout=seconds)
            if thread.is_alive():
                stop_thread.set()
                raise TimeoutError
            return result
        return wrapper
    return decorator

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
        
        @timeout(self.timeout_ms / 1000)
        def iterate():
            for _ in range(self.iterations):
                node = self.tree.select(policy=self.policy, policy_kwargs=self.policy_kwargs)
                value = self.simulate(node)
                self.tree.backpropagate(node, value)
                
        try:
            iterate()
        except TimeoutError as e:
            print(e)
        
        best_child = max(self.tree.root.children, key=lambda child: child.Q)
        return best_child.state.history[-1]
    
class UCT_ADP_Player(UCT_Player):
    def __init__(self, max_depth=10, model=None, epsilon=.0, **kwargs):
        super(UCT_ADP_Player, self).__init__(**kwargs)
    
        self.max_depth = max_depth
        self.model: ValueNetwork = model
        self.sim_policy = PolicyNetwork(epsilon=epsilon)
    
    def simulate(self, node: Node) -> float:
        state = node.state.copy(include_history=True)
        for _ in range(self.max_depth):
            if state.fin():
                break
            
            action = self.sim_policy.forward(state, self.model)
            state.play(action)
            
        if state.fin():
            return state.score()
        
        value = self.model.\
            forward(state).\
            cpu().\
            detach().\
            item()

        return value
    
    def next_move(self, game: Gomoku):
        self.tree = Tree(game, **self.tree_kwargs)
        
        @timeout(self.timeout_ms / 1000)
        def iterate():
            for _ in range(self.iterations):
                node = self.tree.select(policy=self.policy, policy_kwargs=self.policy_kwargs)
                value = self.simulate(node)
                self.tree.backpropagate(node, value)
                
        try:
            iterate()
        except TimeoutError as e:
            print(e)
        
        best_child = max(self.tree.root.children, key=lambda child: child.Q)
        return best_child.state.history[-1]