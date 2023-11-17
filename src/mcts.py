import numpy as np
from typing import TypeVar

Game = TypeVar('Game')

class Node:
    def __init__(self, state: Game, parent=None):
        self.state: Game = state
        self.parent: Node = parent
        self.children: list[Node] = []
        self.n: int = 0
        self.Q: float = .0

    def is_fully_expanded(self) -> bool:
        return len(self.children) and len(self.children) == len(self.state.actions())

    def is_terminal(self) -> bool:
        return self.state.fin()
 
def uct_score(parent: Node, child: Node, **kwargs) -> float:
    C = kwargs.get('C', 1.0)
    exploitation = child.Q / child.n
    exploration = np.sqrt(2 * np.log(parent.n) / child.n)
    return exploitation + C * exploration   

class Tree:
    def __init__(self, state: Game, **kwargs):
        state = state.copy()
        self.root = Node(state)
        self.only_adjacents = kwargs.get('only_adjacents', False)

    def select(self, policy=uct_score, policy_kwargs={}) -> Node:
        node = self.root
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                if not len(node.children):
                    raise Exception("No children")
                node = max(node.children, key=lambda child: policy(node, child, **policy_kwargs))
        return node

    def expand(self, node: Node) -> Node:
        if self.only_adjacents:
            adjacent_actions = node.state.actions(only_adjacents=True)
            adjacent_filtered_actions = list(filter( 
                lambda action: action not in [
                    child.state.history[-1] 
                    for child in node.children
                ], adjacent_actions
            ))
            if len(adjacent_filtered_actions):
                action = adjacent_filtered_actions[np.random.randint(0, len(adjacent_filtered_actions))]
                new_state = node.state.copy(include_history=True)
                new_state.play(action)
                new_node = Node(new_state, parent=node)
                node.children.append(new_node)
                return new_node
                
        actions = list(filter( 
            lambda action: action not in [
                child.state.history[-1] 
                for child in node.children
            ], node.state.actions()
        ))
        if not len(actions):
            raise Exception("No action")
        
        action = actions[np.random.randint(0, len(actions))]
        new_state = node.state.copy(include_history=True)
        new_state.play(action)
        new_node = Node(new_state, parent=node)
        node.children.append(new_node)
        return new_node

    def simulate(self, node: Node) -> float:
        state = node.state.copy(include_history=True)
        while not state.fin():
            state_actions = state.actions()
            if not len(state_actions):
                break
            action = state_actions[np.random.randint(0, len(state_actions))]
            state.play(action)
        return state.score()

    def backpropagate(self, node: Node, reward: float):
        while node is not None:
            node.n += 1
            node.Q += reward
            node = node.parent