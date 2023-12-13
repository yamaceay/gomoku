from typing import Callable
import random

from .players import Player, RandomPlayer
from .gomoku import Gomoku
from .mcts import Node, Tree, uct_score
from .patterns import sortfn
from .adp import ADP_Player

class UCT_Zero_Player(Player):
    def __init__(self, 
                 adp_model: ADP_Player, 
                 max_depth: int = 10,
                 epsilon: float = .25,
                 iterations: int = 400,
                 sim_is_random: bool = False,
                 policy: Callable = uct_score, 
                 policy_kwargs: dict[str] = {}, 
                 tree_kwargs: dict[str] = {}):
        self.iterations = iterations
        self.policy = policy
        self.policy_kwargs = policy_kwargs
        self.tree_kwargs = tree_kwargs
        self.adp = adp_model
        self.max_depth = max_depth
        self.epsilon = epsilon
        self.sim_is_random = sim_is_random
    
    def expand(self, node: Node) -> Node:
        all_actions = node.state.actions()
        other_actions = [
            child.state.last_move
            for child in node.children
        ]
        actions = list(set(all_actions) - set(other_actions))
        assert len(actions), "No action"
        
        action = random.choice(actions)
        new_state = node.state.copy()
        new_state.play(action)
        new_node = Node(new_state, parent=node)
        node.children.append(new_node)
        return new_node
        
    def simulate(self, node: Node) -> float:
        game = node.state.copy()
        for i in range(self.max_depth):
            if game.fin():
                break
            if self.sim_is_random:
                action = RandomPlayer().next_move(game)
            else:
                action = self.adp.next_move(game, epsilon=self.epsilon)
            game.play(action)
        return self.adp(game).cpu().detach().item()

    def rewards_actions(self, game: Gomoku):
        self.tree = Tree(game, **self.tree_kwargs)
        first_player = game.player
        
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
    