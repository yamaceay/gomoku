from .mcts import uct_score
from .players import UCT_Player
from .gomoku import Gomoku

if __name__ == "__main__":
    game_kwargs = {
        'M': 5,
        'N': 5,
        'K': 4,
        'ADJ': 2,
    }
    
    value_network_kwargs = {
        'alpha': 0.9,
        'magnify': 2,
        'gamma': 0.9,
        'lr': 0.01,
        'n_steps': 1,
    }
    
    policy_network_kwargs = {
        'epsilon': 0.1,
    }
    
    # adp_model = ADP_Player("models_wzlen/best.h5", value_network_kwargs, policy_network_kwargs)
    uct_player = UCT_Player(iterations=5000, policy=uct_score, tree_kwargs={'only_adjacents': True})
    
    game = Gomoku(**game_kwargs)
    
    while not game.fin():
        try:
            action = uct_player.next_move(game)
            game.play(action)
            game.print()
        except TimeoutError as e:
            print(e)