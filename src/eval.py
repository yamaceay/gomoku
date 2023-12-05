from tqdm import tqdm
from .zero import AlphaZeroPlayer
from .adp import ADP_Player, ADP_Dense_Player, ADP_Conv_Player
from .players import Player
from .gomoku import Gomoku

from torch.optim import lr_scheduler
import os
import logging
import random

def comp_models(game_kwargs, model1: Player, model2: Player, print_game: bool = False) -> tuple[int, bool, int]:
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
        print(game)
        
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
    epochs_start: int = 0, 
    n_test_games: int = 0, 
    select_best: bool = False,
    eval: bool = True, 
    train: bool = True,
    zero_play: bool = True,
    player: ADP_Player = ADP_Dense_Player,
    player_args: dict = {},
    end_factor: float = 0.5,
    DIR_PATH: str = None,
):
    logger = player_args.get("logger", logging.getLogger(__name__))
    
    BEST_MODEL_PATH = os.path.join(DIR_PATH, "models/best.h5")
    ZERO_RESULTS_PATH = os.path.join(DIR_PATH, "logs/zero_results.log")
    
    len_histories = []
    if not os.path.exists(ZERO_RESULTS_PATH):
        with open(ZERO_RESULTS_PATH, "w") as f:
            f.write("")
        
    with open(ZERO_RESULTS_PATH, "r") as f:
        for line in f.readlines():
            batch, avg_len_history = line.split(",")
            batch = int(batch)
            avg_len_history = float(avg_len_history)
            len_histories += [(avg_len_history, batch)]
    max_len_history = max(len_histories, key=lambda x: x[0]) if len(len_histories) else None

    adp_model = player(model_path=BEST_MODEL_PATH, **player_args)
    
    scheduler = lr_scheduler.LinearLR(adp_model.nn.optimizer, start_factor=1, end_factor=end_factor, total_iters=30)
    for batch in tqdm(range(epochs_start, epochs_end, epochs_step), position=0, leave=False, desc="Batches"):
        last_epoch_in_batch = batch + epochs_step
        new_path = os.path.join(DIR_PATH, "models/epoch_{}.h5".format(last_epoch_in_batch))
        
        if train:
            for i in tqdm(range(1, epochs_step+1), position=1, leave=False, desc="Epochs"):
                game = Gomoku(**game_kwargs)
                if zero_play:
                    loss = adp_model.train_by_zero(game)
                else:
                    loss = adp_model.train(game)
                logger.info("Epoch {}, Loss {}, Lr: {:.8f}".format(batch + i, loss, scheduler.get_last_lr()[-1]))
                scheduler.step()
            adp_model.nn.save_model(new_path)
            
        if eval:
            curr_model = player(model_path=new_path, **player_args)
            avg_len_history = eval_by_zero(
                game_kwargs=game_kwargs,
                curr_model=curr_model,
                n_test_games=n_test_games
            )
            
            new_len_history = (avg_len_history, last_epoch_in_batch)
            
            with open(ZERO_RESULTS_PATH, "a") as f:
                f.write("{},{}\n".format(new_len_history[1], new_len_history[0]))
            
            if select_best:
                if max_len_history is None or max_len_history[0] < new_len_history[0]:
                    max_len_history = new_len_history
                    adp_model.nn.save_model(BEST_MODEL_PATH)
                    logger.info("{} is saved as the strongest model".format(new_path))
                    
                else:
                    old_path = os.path.join(DIR_PATH, "models/epoch_{}.h5".format(max_len_history[1]))
                    adp_model.nn.load_model(old_path)
                    logger.info("{} is loaded as the strongest model".format(old_path))
            else:
                adp_model.nn.save_model(BEST_MODEL_PATH)
                logger.info("{} is saved as the latest model".format(new_path))