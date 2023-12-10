from .players import Player
from .gomoku import Gomoku
from .mcts import Node, Tree, uct_score
from .patterns import sortfn
from .adp import ADP_Player

class UCT_Zero_Player(Player):
    def __init__(self, adp_model: ADP_Player, iterations=10000, policy=uct_score, policy_kwargs={}, tree_kwargs={}):
        self.iterations = iterations
        self.policy = policy
        self.policy_kwargs = policy_kwargs
        self.tree_kwargs = tree_kwargs
        self.adp = adp_model
        
    def simulate(self, node: Node) -> float:
        state = node.state.copy()
        while not state.fin():
            action = self.adp.next_move_probs(state, best=False)
            state.play(action)
        return state.score()

    def rewards_actions(self, game: Gomoku):
        self.tree = Tree(game, **self.tree_kwargs)
        first_player = self.tree.root.state.player
        
        for _ in range(self.iterations):
            node = self.tree.select(policy=self.policy, policy_kwargs=self.policy_kwargs)
            if not node.is_fully_expanded() and not node.is_terminal():
                node = self.tree.expand(node)
            value = self.simulate(node)
            self.tree.backpropagate(node, value)
        
        get_reward = lambda child: uct_score(self.tree.root, child, C=0) * first_player
        get_action = lambda child: child.state.last_move 
        get_reward_action = lambda child: (get_reward(child), get_action(child))
        rewards_actions = map(get_reward_action, self.tree.root.children)
        return sortfn(rewards_actions, key=lambda x: x[0])