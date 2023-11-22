from tqdm import tqdm
from .zero import AlphaZeroPlayer
from .adp import ValueNetwork, PolicyNetwork
import random
import logging
import os
from .gomoku import Gomoku

NAME_OF_TRAINING = "1step"
DIR_PATH = "./models_{}".format(NAME_OF_TRAINING)

# configure a logger which logs to the 'adp.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
file_handler = logging.FileHandler('adp_{}.log'.format(NAME_OF_TRAINING))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def comp_models(game_kwargs, model1: ValueNetwork, model2: ValueNetwork, policy: PolicyNetwork, print_game: bool = False):
    model2_starts = random.random() < .5
    if model2_starts:
        model1, model2 = model2, model1
        
    game = Gomoku(**game_kwargs)
    while not game.fin():
        action = policy.forward(game, model1)
        game.play(action)
        if game.fin():
            break
        action = policy.forward(game, model2)
        game.play(action)
    win = game.score()
    if print_game:
        game.print()
    return win, not model2_starts
      
def train_adp(epochs: int, checkpoint: int, game_kwargs, value_network_kwargs, policy_network_kwargs, epochs_start: int = 0, n_test_games: int = 0, eval: bool = True):
    policy = PolicyNetwork(**policy_network_kwargs)
    
    current_model = ValueNetwork(**value_network_kwargs)
    try:
        current_model.load_model()
    except Exception as e:
        logger.info(e)
        
    for i in tqdm(range(epochs_start+1, epochs+1), position=0, leave=False, desc="Training"):
        game = Gomoku(**game_kwargs)
        loss = current_model.train(game, policy)
        logger.info("Epoch {}, Loss {}".format(i, loss))
        
        if i % checkpoint == 0:
            new_path = os.path.join(DIR_PATH, "epoch_%s.h5" % i)
            current_model.save_model(new_path)
            
            if eval:
                best_model = AlphaZeroPlayer(**game_kwargs)
                
                curr_model_stats = []
                best_model_stats = []
                for _ in tqdm(range(n_test_games), position=1, leave=False, disable=(not eval), desc="Testing epoch {}".format(i)):
                    win, curr_model_starts = comp_models(game_kwargs, current_model, best_model, policy)
                    if curr_model_starts:
                        curr_model_stats += [int(win > 0)]
                    else:
                        best_model_stats += [int(win <= 0)]
                
                model_stats = curr_model_stats + best_model_stats
                
                logger.info("As 1st player, current model won {} of {} games".format(sum(curr_model_stats), len(curr_model_stats)))
                logger.info("As 2nd player, current model won {} of {} games".format(sum(best_model_stats), len(best_model_stats)))
                logger.info("Avg. win rate of current model: {}".format(sum(model_stats) / len(model_stats)))
                
                if sum(model_stats) > n_test_games / 2:    
                    current_model.save_model()  
                else:
                    current_model.load_model() 

if __name__ == "__main__":
    train_adp(
        # epochs_start = 20,
        epochs = 200, 
        checkpoint = 10, 
        # n_test_games = 9, 
        eval = False, 
        game_kwargs={
            'M': 8,
            'N': 8,
            'K': 5,
            'ADJ': 2,
        }, value_network_kwargs={
            'alpha': 0.9,
            'magnify': 2,
            'gamma': 0.9,
            'lr': 0.01,
            'n_steps': 1,
        }, policy_network_kwargs={
            'epsilon': 0.1,
        }
    )
    
    # print(comp_models(
    #     {'M': 7, 'N': 7, 'K': 5, 'ADJ': 2},
    #     ValueNetwork(alpha=0.9, magnify=1, model_path="epoch_5.h5"), 
    #     ValueNetwork(alpha=0.9, magnify=1, model_path="epoch_20.h5"), 
    #     policy=PolicyNetwork(epsilon = 0.1),
    #     print_game=True,
    # ))