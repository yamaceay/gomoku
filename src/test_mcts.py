from .mcts import uct_score, uct_pb_score
from .players import UCT_Player
from .adp import ADP_Conv_Player
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
    
    adp_model = ADP_Conv_Player(
        model_path="models_fwd/best.h5", 
        M=game_kwargs['M'],
        N=game_kwargs['N'],
        **value_network_kwargs, 
        **policy_network_kwargs,
    )
    uct_player = UCT_Player(
        iterations=1000, 
        policy=uct_score,
    )
    
    uct_player_2 = UCT_Player(
        iterations=1000, 
        policy=uct_pb_score,
    )
    
    game = Gomoku(**game_kwargs)
    
    while not game.fin():
        action = uct_player.next_move(game)
        game.play(action)
        print(game)
        if game.fin():
            break
        action = uct_player_2.next_move(game)
        game.play(action)
        print(game)