from typing import Callable
import numpy as np
from .patterns import PB_DICT_5, sortfn, pb_heuristic
from .gomoku import Gomoku
from .players import Player, RandomPlayer

class Node:
    def __init__(self, state: Gomoku, parent=None):
        self.state: Gomoku = state
        self.parent: Node = parent
        self.children: list[Node] = []
        self.n: int = 0
        self.Q: float = .0
        self.depth: int = 0 if parent is None else parent.depth + 1

    def is_fully_expanded(self) -> bool:
        moves = self.state.actions()
        return len(self.children) and len(self.children) == len(moves)

    def is_terminal(self) -> bool:
        return self.state.fin()
    
    def __repr__(self):
        history = self.state.history()
        if len(history):
            return f"Node({history}, {self.Q:.2f} / {self.n})"
        else:
            return f"Node({self.Q:.2f} / {self.n})"

def uct_score(parent: Node, child: Node, **kwargs) -> float:
    C = kwargs.get('C', 1.0)
    exploitation = child.Q / child.n
    exploration = np.sqrt(2 * np.log(parent.n) / child.n)
    return exploitation + C * exploration   

def pb_fn(game: Gomoku) -> float:
    pb_curr = game.find_patterns()
    
    pb_value = 0
    for pattern in pb_curr:
        pattern_score = pb_heuristic(PB_DICT_5[pattern])
        [curr_x, curr_o] = pb_curr.get(pattern, [0, 0])
        pb_value += (curr_x - curr_o) * pattern_score
    
    pb_value *= -game.player
    return pb_value

def pb_score(parent: Node, child: Node) -> float:
    move = child.state.last_move
    pb_parent = parent.state.find_patterns(move)
    pb_child = child.state.find_patterns(move)
        
    pb_value = 0
    for pattern in pb_parent | pb_child:
        pattern_score = pb_heuristic(PB_DICT_5[pattern])
        
        [parent_x, parent_o] = pb_parent.get(pattern, [0, 0])
        [child_x, child_o] = pb_child.get(pattern, [0, 0])
        
        child_score = child_x - child_o
        parent_score = parent_x - parent_o
        
        pb_value += (child_score - parent_score) * pattern_score
        
    pb_value *= -child.state.player
    
    return pb_value
    
def uct_pb_score(
    parent: Node, 
    child: Node, 
    C: float = 1,
    C_PB: float = 5) -> float:
    
    ucb = uct_score(parent, child, C=C)
    pbs = pb_score(parent, child)
    return ucb + C_PB * pbs

class Tree:
    def __init__(self, 
                 state: Gomoku, 
                 gamma: float = 0.9,
                 value_fn: Callable = None,
                 max_depth: int = None):
        self.state = state.copy()
        self.root = Node(self.state)
        self.gamma = gamma
        self.value_fn = value_fn
        self.max_depth = max_depth if max_depth is not None else state.M * state.N

    def select(self, policy: Callable = uct_score, policy_kwargs: dict = {}) -> Node:
        node = self.root
        while not node.is_terminal() and node.is_fully_expanded():
            assert len(node.children), "No children"
            sort_key = lambda child: policy(node, child, **policy_kwargs)
            child_nodes = sortfn(node.children, sort_key)
            node = child_nodes[0]
        return node

    def expand(self, node: Node) -> Node:
        actions = [
            action 
            for action in node.state.actions() 
            if action not in [
                child.state.last_move
                for child in node.children
            ]
        ]
        assert len(actions), "No action"
        
        action = actions[np.random.randint(0, len(actions))]
        new_state = node.state.copy()
        new_state.play(action)
        new_node = Node(new_state, parent=node)
        node.children.append(new_node)
        return new_node

    def simulate(self, node: Node) -> float:
        state = node.state.copy()
        player = RandomPlayer()
        for _ in range(self.max_depth):
            if state.fin():
                return state.score()
            action = player.next_move(state)
            state.play(action)
        return self.value_fn(state)
    
    def backpropagate(self, node: Node, reward: float):
        reward *= -node.state.player
        while node is not None:
            node.n += 1
            node.Q += reward
            reward *= -self.gamma
            node = node.parent
            
class UCT_Player(Player):
    def __init__(self, 
                 iterations=10000, 
                 policy=uct_score, 
                 policy_kwargs={},
                 tree_kwargs={}):
        self.iterations = iterations
        self.policy = policy
        self.policy_kwargs = policy_kwargs
        self.tree_kwargs = tree_kwargs

    def simulate(self, node: Node) -> float:
        return self.tree.simulate(node)

    def expand(self, node: Node) -> Node:
        return self.tree.expand(node)

    def rewards_actions(self, game: Gomoku):
        self.tree = Tree(game, **self.tree_kwargs)
        first_player = self.tree.root.state.player
        
        for _ in range(self.iterations):
            node = self.tree.select(policy=self.policy, policy_kwargs=self.policy_kwargs)
            if not node.is_fully_expanded() and not node.is_terminal():
                node = self.expand(node)
            value = self.simulate(node)
            self.tree.backpropagate(node, value)
        
        get_reward = lambda child: uct_score(self.tree.root, child, C=0) * first_player
        get_action = lambda child: child.state.last_move 
        get_reward_action = lambda child: (get_reward(child), get_action(child))
        rewards_actions = map(get_reward_action, self.tree.root.children)
        return sortfn(rewards_actions, key=lambda x: x[0])

if __name__ == '__main__':
    uct = UCT_Player(
        iterations=10000, 
        policy=uct_score,
        policy_kwargs={
            "C": 1,
        },
        tree_kwargs={
            "max_depth": 10,
            "value_fn": pb_fn,
        }
    )
    
    game_kwargs = {
        "M": 8,
        "N": 8,
        "K": 5,
        "ADJ": 2,
    }
    
    game = Gomoku(**game_kwargs)
    while not game.fin():
        action = uct.next_move(game)
        game.play(action)
        print(game)