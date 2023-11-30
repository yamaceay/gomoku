from .adp import ADP_Player, ADP_Value_Net
from .zero import AlphaZeroPlayer
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
    
    dir_name = "models_wzno"
    model_path = f"{dir_name}/epoch_1000.h5"
    
    adp = ADP_Player(model_path=model_path, value_network_kwargs=value_network_kwargs, policy_network_kwargs=policy_network_kwargs)
    
    game = Gomoku(**game_kwargs)

    while not game.fin():
        value_network = ADP_Value_Net(model_path=model_path, **value_network_kwargs)
        rewards_actions = value_network.get_rewards_actions(game)
        print(rewards_actions[:5], rewards_actions[-5:])
        move = adp.next_move(game)
        game.play(move)
        game.print()