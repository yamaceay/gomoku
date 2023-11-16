from .gomoku import Gomoku
from .mcts import Tree, Node, uct_score
import numpy as np
import threading

TIMEOUT = 2.5

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
            
class UCTPlayer(Player):
    def __init__(self, iterations=1000, policy=uct_score, policy_kwargs={}):
        self.iterations = iterations
        self.policy = policy
        self.policy_kwargs = policy_kwargs

    def next_move(self, game: Gomoku):
        tree = Tree(game)
        
        @timeout(TIMEOUT)
        def iterate():
            for _ in range(self.iterations):
                node = tree.select(policy=self.policy, policy_kwargs=self.policy_kwargs)
                value = tree.simulate(node)
                tree.backpropagate(node, value)
                
        try:
            iterate()
        except TimeoutError as e:
            print(e)
        
        best_child = max(tree.root.children, key=lambda child: child.Q)
        return best_child.state.history[-1]
    
class Model:
    def __call__(self, state: Gomoku) -> float:
        raise NotImplementedError
    
class ADPModel(Model):
    def __call__(self, state: Gomoku) -> float:
        return np.random.random()

line_to_int = lambda k: [1 if c == 'x' else -1 if c == 'o' else 0 for c in k]
int_to_line = lambda k: ''.join(['x' if c == 1 else 'o' if c == -1 else '-' for c in k])   
    
def pb_score(parent: Node, child: Node, **kwargs) -> float:
    decay = kwargs.get('decay', 0.9)
    PB_DICT = {
        '-oooo-': 10000,
        'x-ooo-x': 1000*decay**3,
        'x-ooo--': 1000*decay**1,
        '--ooo--': 1000,
        '--ooo-o': 1000,
        'o-ooo-o': 10000,
        'x-ooo-o': 1000*decay**1,
        '--oo--': 100,
        'x-oo--': 100*decay**1,
        'xoo-o--': 100*decay**2,
        'xo-oo--': 100*decay**2,
        'xo-oo-x': 100*decay**3,
        'xoo-o-x': 100*decay**3,
        'xooo--':  100,
        'xoo---': 1,
        'xoooo-': 1000,
        '-oo-o-': 1000*decay**1,
        'xooo-o-': 1000*decay**1,
        'xoo-oo-': 1000*decay**1,
        'xooo-ox': 1000*decay**3,
        'xoo-oox': 1000*decay**3,
        'xooo-oo': 1000*decay**1,
        '-o-o-o-': 1000*decay**2,
        'xo-o-ox': 1000*decay**6,
        'xo-o-o-': 1000*decay**4,
        '--o-o--': 100*decay**2,
        'x-o-o--': 100*decay**4,
        'x-o-o-x': 100*decay**6,
        '--o--': 1,
    }
    
    move = child.state.history[-1]
    pb_value = 0
    for pattern, value in PB_DICT.items():
        values = line_to_int(pattern)
        pb_parent = parent.state.find_near(move, values)
        pb_child = child.state.find_near(move, values)
        pb_diff = (pb_child - pb_parent) * value
        
        reverse_values = [-c for c in values]
        pb_reverse_parent = parent.state.find_near(move, reverse_values)
        pb_reverse_child = child.state.find_near(move, reverse_values)
        pb_reverse_diff = (pb_reverse_child - pb_reverse_parent) * (-5 * value)
        
        pb_value += pb_diff + pb_reverse_diff
    pb_value *= child.state.player
    pb_value /= 100000
    return pb_value
    
def uct_pb_score(parent: Node, child: Node, **kwargs) -> float:
    C_PB = kwargs.get('C_PB', 5)
    ucb = uct_score(parent, child, **kwargs)
    pbs = pb_score(parent, child, **kwargs)
    return ucb + C_PB * pbs
    
class UCTQPlayer(Player):
    def __init__(self, iterations=1000, max_depth=10, policy=uct_score, policy_kwargs={}, model=None):
        self.iterations = iterations
        self.max_depth = max_depth
        self.policy = policy
        self.policy_kwargs = policy_kwargs
        self.model = model
    
    def simulate(self, node: Node):
        state = node.state.copy(include_history=True)
        for _ in range(self.max_depth):
            if not state.fin():
                state_actions = state.actions()
                if not len(state_actions):
                    break
                action = state_actions[np.random.randint(0, len(state_actions))]
                state.play(action)
            else:
                if self.model is not None:
                    return self.model(state)
                else:
                    return np.random.random()
        return state.score()
        
    def next_move(self, game: Gomoku):
        tree = Tree(game)
        
        @timeout(TIMEOUT)
        def iterate():
            for _ in range(self.iterations):
                node = tree.select(policy=self.policy, policy_kwargs=self.policy_kwargs)
                value = tree.simulate(node)
                tree.backpropagate(node, value)
        
        try:
            iterate()
        except TimeoutError as e:
            print(e)
        
        best_child = max(tree.root.children, key=lambda child: child.Q)
        return best_child.state.history[-1]