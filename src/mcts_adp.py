from typing import Callable
from collections import deque

from .players import Player
from .gomoku import Gomoku
from .mcts import Node, Tree, uct_score
from .patterns import sortfn
from .adp import ADP_Player
from .data import collect_play_data
import random
from tqdm import tqdm

class UCT_Zero_Player(Player):
    def __init__(self, 
                 adp_model: ADP_Player, 
                 iterations: int = 10000, 
                 policy: Callable = uct_score, 
                 policy_kwargs: dict[str] = {}, 
                 tree_kwargs: dict[str] = {}):
        self.iterations = iterations
        self.policy = policy
        self.policy_kwargs = policy_kwargs
        self.tree_kwargs = tree_kwargs
        self.adp = adp_model

    # def train(self, 
    #               game_kwargs: dict[str, int],
    #               buffer_size: int = 1000,
    #               batch_size: int = 200,
    #               iterations: int = 25,
    #               epsilon: float = .1):
    #     losses = []
    #     buffer = deque(maxlen=buffer_size)
    #     game = Gomoku(**game_kwargs)
        
    #     learner_args = {
    #         "player1": self.adp,
    #         "epsilon1": epsilon,
    #     }
        
    #     with tqdm(desc="Training", leave=False, position=1) as pbar:
    #         while not game.fin():
    #             self.tree = Tree(game, **self.tree_kwargs)
                
    #             for _ in range(iterations):
    #                 node = self.tree.select(policy=self.policy, policy_kwargs=self.policy_kwargs)
    #                 if not node.is_fully_expanded() and not node.is_terminal():
    #                     node = self.tree.expand(node)
    #                 game, _ = play_until_end(node.state, **learner_args)
    #                 buffer.extend(collect_play_data(game))
    #                 self.tree.backpropagate(node, game.score())
                
    #             batch = random.sample(buffer, min(batch_size, len(buffer)))
                
    #             loss = self.adp.train_batch(batch, disable=False)
    #             losses += [loss]
    #             pbar.set_postfix(loss=loss)
    #             pbar.update()
        
    #     return losses
    
    def expand(self, node: Node) -> Node:
        new_state = node.state.copy()
        action = self.adp.next_move(new_state)
        new_state.play(action)
        new_node = Node(new_state, parent=node)
        node.children.append(new_node)
        return new_node
        
    def simulate(self, node: Node) -> float:
        return self.adp(node.state).cpu().detach().item()

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
    from .adp import ADP_Pre_Player
    
    buffer = deque(maxlen=1000)
    
    game_kwargs = {
        "M": 8,
        "N": 8,
        "K": 5,
        "ADJ": 2,
    }
    
    adp = ADP_Pre_Player(game_kwargs=game_kwargs)
    
    # uct = UCT_Player(iterations=2000, policy=uct_score)
    # zero = AlphaZeroPlayer(game_kwargs)
    
    # depth = 8
    # for i in range(50):
    #     game = Gomoku(**game_kwargs)
    #     for i in tqdm(range(game.M * game.N)):                        
    #         if i < depth:
    #             # if i % 2 == 0:
    #             #     print("ZERO BEGIN")
    #             #     action = zero.next_move(game, epsilon=.25)
    #             #     print("ZERO END")
    #             # else:
    #             print("ADP BEGIN")
    #             action = adp.next_move(game, epsilon=.1)
    #             print("ADP END")
            
    #         else:
    #             print("UCT BEGIN")
    #             rewards_actions = uct.rewards_actions(game)
    #             print("UCT END")

    #             if game.player == -1:
    #                 _, action = rewards_actions[-1]
    #             else:
    #                 _, action = rewards_actions[0]
                    
    #             print("COLLECT BEGIN")
    #             for r, a in rewards_actions:
    #                 new_game = game.copy()
    #                 new_game.play(a)
    #                 buffer.extend([(s, r) for s, _ in collect_play_data(new_game)])
    #             print("COLLECT END", len(buffer))
                
    #             print("TRAIN BEGIN")
    #             sample = random.sample(buffer, min(400, len(buffer)))
    #             loss = adp.train_batch(sample, disable=False)
    #             print("TRAIN END", loss)
            
    #         game.play(action)
    #         print(game)
    #         if game.fin():
    #             break    
            
    #     adp.nn.save_model(f"_dens3/models/epoch_{i}.h5")