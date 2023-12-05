import logging
import os
from src import train_adp, ADP_Dense_Player

DIR_PATH = '_conv'

LOSSES_PATH = os.path.join(DIR_PATH, "logs/losses.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
file_handler = logging.FileHandler(LOSSES_PATH)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
        'logger': logger,
    }
    
    policy_network_kwargs = {
        'epsilon': 0.1,
    }
    
    train_adp(
        epochs_start = 0,
        epochs_end = 20, 
        epochs_step = 5, 
        eval=True,
        train=True,
        zero_play=False,
        n_test_games=7,
        select_best = False,
        game_kwargs=game_kwargs, 
        player=ADP_Dense_Player,
        player_args={
            # 'M': game_kwargs['M'],
            # 'N': game_kwargs['N'],
            **value_network_kwargs, 
            **policy_network_kwargs,
        },
        end_factor=0.1,
        DIR_PATH=DIR_PATH,
    )
            