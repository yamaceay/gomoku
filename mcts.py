import numpy as np
from typing import TypeVar

Game = TypeVar('Game')
C = 1

class Node:
    def __init__(self, state: Game, parent=None):
        self.state: Game = state
        self.parent: Node = parent
        self.children: list[Node] = []
        self.n: int = 0
        self.Q: float = .0

    def is_fully_expanded(self):
        return len(self.children) and len(self.children) == len(self.state.actions())

    def is_terminal(self):
        return self.state.fin()

    def uct_score(self, child):
        exploitation = child.Q / child.n
        exploration = np.sqrt(2 * np.log(self.n) / child.n)
        return exploitation + C * exploration   
 
class Tree:
    def __init__(self, state: Game):
        state = state.copy()
        self.root = Node(state)

    def select(self):
        node = self.root
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                if not len(node.children):
                    raise Exception("No children")
                node = max(node.children, key=lambda child: node.uct_score(child))
        return node

    def expand(self, node: Node):
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

    def simulate(self, node: Node):
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