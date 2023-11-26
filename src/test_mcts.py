from .mcts import uct_score
from .adp import ADP_Player, UCT_ADP_Player
from .gomoku import Gomoku

if __name__ == "__main__":
    game_kwargs = {
        'M': 8,
        'N': 8,
        'K': 5,
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
    
    adp_model = ADP_Player("models_wzlen/best.h5", value_network_kwargs, policy_network_kwargs)
    uct_adp_player = UCT_ADP_Player(timeout_ms=1000, max_depth=10, policy=uct_score, model=adp_model, iterations=10000)
    
    game = Gomoku(**game_kwargs)
    
    try:
        move = uct_adp_player.next_move(game)
        game.play(move)
        game.print()
    except TimeoutError as e:
        print(e)