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
                 iterations: int = 10000, 
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
            
        raise Exception("No action")
        
    def simulate(self, node: Node) -> float:
        game = node.state.copy()
        for i in range(self.max_depth):
            if game.fin():
                break
            action = RandomPlayer().next_move(game)
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
    
if __name__ == '__main__':
    from .adp import ADP_Dense_Player, ADP_Pre_Player
    
    game_kwargs = {
        "M": 8,
        "N": 8,
        "K": 5,
        "ADJ": 2,
    }
    
    # adp = ADP_Dense_Player(model_path="_dens2/models/epoch_1000.h5", game_kwargs=game_kwargs)
    adp = ADP_Dense_Player(model_path="_dens/models/epoch_1000.h5", game_kwargs=game_kwargs)
    
    uct_adp = UCT_Zero_Player(adp, iterations=200, policy_kwargs={"C": .1})
    
    game = Gomoku(**game_kwargs)
    while not game.fin():
        rewards_actions = uct_adp.rewards_actions(game)
        print("\n".join([f"{a} {r:.4f}" for r, a in rewards_actions]))

        action = uct_adp.next_move(game, epsilon=.25)
        game.play(action)
        print(game)
        
    # results = eval_by_uct(game_kwargs, adp, adp, n_test_games=3, iterations=200, epsilon=.1)
    # print(results)
    
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
