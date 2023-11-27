from tqdm import tqdm
from .zero import AlphaZeroPlayer
from .adp import ADP_Player, ValueNetwork, PolicyNetwork
from .players import Player
import random
import logging
import os
from .gomoku import Gomoku

NAME_OF_TRAINING = "wzlen"
DIR_PATH = "./models_{}".format(NAME_OF_TRAINING)

# configure a logger which logs to the 'adp.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
file_handler = logging.FileHandler('logs/adp_{}.log'.format(NAME_OF_TRAINING))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def comp_models(game_kwargs, model1: Player, model2: Player, print_game: bool = False) -> tuple[bool, bool, int]:
    game = Gomoku(**game_kwargs)
    
    model2_starts = random.random() < .5
    if model2_starts:
        model1, model2 = model2, model1
        
    while not game.fin():
        action = model1.next_move(game)
        game.play(action)
        if game.fin():
            break
        action = model2.next_move(game)
        game.play(action)
    
    win = game.score()
    if print_game:
        game.print()
        
    return win, not model2_starts, len(game.get_history())
    
def eval_by_zero(game_kwargs, curr_model, n_test_games: int):
    zero = AlphaZeroPlayer(**game_kwargs)
    len_histories = []
    for _ in tqdm(range(n_test_games), position=1, leave=False, desc="Testing"):
        win, curr_model_starts, len_history = comp_models(game_kwargs, curr_model, zero)
        zero.restart()
        curr_model_won = int((win > 0) == curr_model_starts)
        if curr_model_won:
            print("Current model won against zero, amazing!!")
            return game_kwargs['M'] * game_kwargs['N']
        len_histories += [len_history]
    avg_len_history = sum(len_histories) / len(len_histories)
    return avg_len_history
      
def train_adp(
    epochs_end: int, 
    epochs_step: int,
    game_kwargs, 
    model_path: str, 
    value_network_kwargs, 
    policy_network_kwargs, 
    epochs_start: int = 0, 
    n_test_games: int = 0, 
    eval: bool = True, 
    train: bool = True,
    zero_play: bool = True,
):
    
    len_histories = []
    with open("logs/len_histories_{}.txt".format(NAME_OF_TRAINING), "r") as f:
        for line in f.readlines():
            batch, avg_len_history = line.split(",")
            batch = int(batch)
            avg_len_history = float(avg_len_history)
            len_histories += [(avg_len_history, batch)]
    max_len_history = max(len_histories, key=lambda x: x[0]) if len(len_histories) else None

    value_network = ValueNetwork(model_path=model_path, **value_network_kwargs)
    
    for batch in tqdm(range(epochs_start, epochs_end, epochs_step), position=0, leave=False, desc="Batches"):
        last_epoch_in_batch = batch + epochs_step
        new_path = os.path.join(DIR_PATH, "epoch_{}.h5".format(last_epoch_in_batch))
        
        if train:
            policy = PolicyNetwork(**policy_network_kwargs)
            for i in tqdm(range(1, epochs_step+1), position=1, leave=False, desc="Epochs"):
                game = Gomoku(**game_kwargs)
                if zero_play:
                    loss = value_network.train_by_zero(game, policy)
                else:
                    loss = value_network.train(game, policy)
                logger.info("Epoch {}, Loss {}".format(batch + i, loss))
            value_network.save_model(new_path)
            
        if eval:
            curr_model = ADP_Player(new_path, value_network_kwargs, policy_network_kwargs)
            avg_len_history = eval_by_zero(
                game_kwargs=game_kwargs,
                curr_model=curr_model,
                n_test_games=n_test_games
            )
            
            new_len_history = (avg_len_history, last_epoch_in_batch)
            
            with open("logs/len_histories_{}.txt".format(NAME_OF_TRAINING), "a") as f:
                f.write("{},{}\n".format(new_len_history[1], new_len_history[0]))
            
            if max_len_history is None or max_len_history[0] < new_len_history[0]:
                max_len_history = new_len_history
                value_network.save_model(model_path)
                logger.info("{} is saved as the strongest model".format(new_path))
                
            else:
                old_path = os.path.join(DIR_PATH, "epoch_{}.h5".format(max_len_history[1]))
                value_network.load_model(old_path)
                logger.info("{} is loaded as the strongest model".format(old_path))
                
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
        epochs_start = 10,
        epochs_end = 50, 
        epochs_step = 10, 
        eval=True,
        train=True,
        zero_play=True,
        n_test_games=7,
        game_kwargs = game_kwargs, 
        model_path = os.path.join(DIR_PATH, 'best.h5'),
        value_network_kwargs = value_network_kwargs,
        policy_network_kwargs = policy_network_kwargs,
    )
            