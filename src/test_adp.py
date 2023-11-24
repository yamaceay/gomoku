from .adp import ADP_Player, get_rewards_actions, ValueNetwork
from .gomoku import Gomoku
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':
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
        'logger': logger,
    }
    
    policy_network_kwargs = {
        'epsilon': 0.1,
    }
    
    model_path = "models_1step_wzero/best.h5"
    
    adp = ADP_Player(model_path=model_path, value_network_kwargs=value_network_kwargs, policy_network_kwargs=policy_network_kwargs)
    
    game = Gomoku(**game_kwargs)

    while not game.fin():
        value_network = ValueNetwork(model_path=model_path, **value_network_kwargs)
        print(get_rewards_actions(game, value_network))
        move = adp.next_move(game)
        game.play(move)
        game.print()