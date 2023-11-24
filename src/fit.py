from tqdm import tqdm
from .zero import AlphaZeroPlayer
from .adp import ADP_Player, ValueNetwork, PolicyNetwork
from .players import Player
import random
import logging
import os
from .gomoku import Gomoku

NAME_OF_TRAINING = "wzero"
DIR_PATH = "./models_{}".format(NAME_OF_TRAINING)

# configure a logger which logs to the 'adp.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
file_handler = logging.FileHandler('logs/adp_{}.log'.format(NAME_OF_TRAINING))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def comp_models(game_kwargs, model1: Player, model2: Player, print_game: bool = False):
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
        
    return win, not model2_starts
    
def eval_adp(game_kwargs, curr_model, best_model, n_test_games: int):
    curr_model_stats = []
    best_model_stats = []
    for _ in tqdm(range(n_test_games), position=1, leave=False, disable=(not eval), desc="Testing"):
        win, curr_model_starts = comp_models(game_kwargs, curr_model, best_model)
        if curr_model_starts:
            curr_model_stats += [int(win > 0)]
        else:
            best_model_stats += [int(win <= 0)]
    
    model_stats = curr_model_stats + best_model_stats
    
    # if sum(model_stats) > n_test_games / 2:    
    #     curr_model.value_network.save_model()  
        
    logger.info("As 1st player, current model won {} of {} games".format(sum(curr_model_stats), len(curr_model_stats)))
    logger.info("As 2nd player, current model won {} of {} games".format(sum(best_model_stats), len(best_model_stats)))
    logger.info("Avg. win rate of current model: {:.2f}%".format(sum(model_stats) * 100 / len(model_stats)))
      
def train_adp(
    epochs: int, 
    checkpoint: int,
    game_kwargs, 
    model_path: str, 
    value_network_kwargs, 
    policy_network_kwargs, 
    epochs_start: int = 0, 
    # n_test_games: int = 0, 
    # eval: bool = False, 
    zero_play: bool = True,
):
    policy = PolicyNetwork(**policy_network_kwargs)
    
    value_network = ValueNetwork(model_path=model_path, **value_network_kwargs)
        
    for i in tqdm(range(epochs_start+1, epochs+1), position=0, leave=False, desc="Training"):
        game = Gomoku(**game_kwargs)
        if zero_play:
            loss = value_network.train_by_zero(game, policy)
        else:
            loss = value_network.train(game, policy)
        logger.info("Epoch {}, Loss {}".format(i, loss))
        
        if i % checkpoint == 0:
            new_path = os.path.join(DIR_PATH, "epoch_%s.h5" % i)
            value_network.save_model(new_path)
            
            # if eval:
            #     curr_model = ADP_Player(model_path, value_network_kwargs, policy_network_kwargs)
            #     best_model = AlphaZeroPlayer(**game_kwargs)
                
            #     eval_adp(
            #         game_kwargs=game_kwargs, 
            #         curr_model=curr_model,
            #         best_model=best_model,
            #         n_test_games=n_test_games
            #     )

if __name__ == "__main__":
    epochs_start = 0
    train_instead_of_test = False
    
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
    
    if train_instead_of_test:
        train_adp(
            epochs_start = epochs_start,
            epochs = 100, 
            checkpoint = 10, 
            zero_play=True,
            game_kwargs = game_kwargs, 
            model_path = os.path.join(DIR_PATH, 'best.h5'),
            value_network_kwargs = value_network_kwargs,
            policy_network_kwargs = policy_network_kwargs,
        )
    
    else:
        curr_model = ADP_Player(
            model_path=os.path.join(DIR_PATH, "best.h5"),
            value_network_kwargs=value_network_kwargs, 
            policy_network_kwargs=policy_network_kwargs,
        )
        
        # best_model = AlphaZeroPlayer(**game_kwargs)
        for i in range(1, 40):
            best_model = ADP_Player(
                model_path=os.path.join(DIR_PATH, "epoch_{}.h5".format(i*10 + epochs_start)), 
                value_network_kwargs=value_network_kwargs, 
                policy_network_kwargs=policy_network_kwargs,
            )
            
            eval_adp(
                game_kwargs=game_kwargs,
                curr_model=curr_model,
                best_model=best_model,
                n_test_games=7,
            )
            