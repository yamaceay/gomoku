import numpy as np
from .patterns import PB_DICT, lti, sortfn
from .gomoku import Gomoku

class Node:
    def __init__(self, state: Gomoku, parent=None):
        self.state: Gomoku = state
        self.parent: Node = parent
        self.children: list[Node] = []
        self.n: int = 0
        self.Q: float = .0

    def is_fully_expanded(self) -> bool:
        moves = self.state.actions(only_adjacents=True)
        return len(self.children) and len(self.children) == len(moves)

    def is_terminal(self) -> bool:
        return self.state.fin()
    
    def __repr__(self):
        if len(self.state.history):
            return f"Node({self.state.history}, {self.Q} / {self.n})"
        else:
            return f"Node({self.Q:.2f} / {self.n})"

def uct_score(parent: Node, child: Node, **kwargs) -> float:
    C = kwargs.get('C', 1.0)
    exploitation = child.Q / child.n
    exploration = np.sqrt(2 * np.log(parent.n) / child.n)
    return exploitation + C * exploration   

def pb_score(parent: Node, child: Node, **kwargs) -> float:
    decay = kwargs.get('decay', 0.9)
    
    move = child.state.get_history()[-1]
    pb_value = 0
    for pattern, value_fn in PB_DICT.items():
        pattern_score = value_fn(decay)
        values = list(map(lti, pattern))
        pb_parent = parent.state.find_near(move, values)
        pb_child = child.state.find_near(move, values)
        pb_diff = (pb_child - pb_parent) * pattern_score
        
        reverse_values = [-c for c in values]
        pb_reverse_parent = parent.state.find_near(move, reverse_values)
        pb_reverse_child = child.state.find_near(move, reverse_values)
        pb_reverse_diff = (pb_reverse_child - pb_reverse_parent) * (-5 * pattern_score)
        
        pb_value += pb_diff + pb_reverse_diff
    pb_value *= child.state.player
    pb_value /= 100000
    return pb_value
    
def uct_pb_score(parent: Node, child: Node, **kwargs) -> float:
    C_PB = kwargs.get('C_PB', 5)
    ucb = uct_score(parent, child, **kwargs)
    pbs = pb_score(parent, child, **kwargs)
    return ucb + C_PB * pbs

class Tree:
    def __init__(self, state: Gomoku, **kwargs):
        self.state = state.copy()
        self.root = Node(self.state)
        self.only_adjacents = kwargs.get('only_adjacents', False)
        self.decay = kwargs.get('decay', 0.9)

    def select(self, policy=uct_score, policy_kwargs={}) -> Node:
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
            for action in node.state.actions(only_adjacents=True) 
            if action not in [
                child.state.get_history()[-1] 
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
        while not state.fin():
            state_actions = state.actions(only_adjacents=True)
            assert len(state_actions), "No action"
            action = state_actions[np.random.randint(0, len(state_actions))]
            state.play(action)
        return state.score()
    
    def backpropagate(self, node: Node, reward: float):
        reward *= -node.state.player
        while node is not None:
            node.n += 1
            node.Q += reward
            reward *= -self.decay
            node = node.parent