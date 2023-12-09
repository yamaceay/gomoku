from tqdm import tqdm
from .zero import AlphaZeroPlayer
from .adp import ADP_Player
from .players import Player
from .gomoku import Gomoku
from .data import collect_selfplay_data, collect_play_data, play_until_end

from collections import deque
from torch.optim import lr_scheduler
import os
import logging
import random

def comp_models(game_kwargs: dict[int],
    player1: Player,
    player2: Player,
    epsilon1: float = .0,
    epsilon2: float = .0) -> list[tuple[str, bool]]:
        
    learner_args = {
        "player1": player1,
        "epsilon1": epsilon1,
    }
    
    trainer_args = {
        "player1": player2,
        "epsilon1": epsilon2,
    }
    
    game, learner_starts = play_until_end(game_kwargs, **learner_args, **trainer_args)
    
    return game.score(), learner_starts, len(game.history())

def tournament(game_kwargs, models: list[Player], n_test_games: int, start_ind: int = 0) -> list[tuple[int, int, int, int]]:
    total_ind = 0
    with tqdm(
        total=n_test_games * len(models) * (len(models) - 1) // 2, 
        position=0, 
        leave=False, 
        desc="Tournament"
    ) as bar:
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                if total_ind < start_ind:
                    bar.update(n_test_games)
                    total_ind += n_test_games
                    continue
                n_wins, l_history = 0, 0
                for _ in range(n_test_games):
                    win, starts, len_history = comp_models(game_kwargs, models[i], models[j])
                    n_wins += (win > 0) == starts
                    l_history += len_history
                    bar.update(1)
                    total_ind += 1
                n_wins, l_history = n_wins / n_test_games, l_history / n_test_games
                yield (i, j, n_wins, l_history)

def eval_by_zero(game_kwargs: dict[int], 
                 curr_model: ADP_Player, 
                 zero_model: AlphaZeroPlayer, 
                 n_test_games: int, 
                 epsilon: float = .1):
    len_histories = []
    for _ in tqdm(range(n_test_games), position=1, leave=False, desc="Testing"):
        win, curr_model_starts, len_history = comp_models(game_kwargs, curr_model, zero_model, epsilon=epsilon)
        zero_model.restart()
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
    player: ADP_Player,
    epochs_start: int = 0, 
    n_test_games: int = 0, 
    select_best: bool = False,
    eval: bool = True, 
    train: bool = True,
    zero_play: bool = True,
    player_args: dict = {},
    lr_args: dict = {},
    epsilon: float = .1,
    buffer_size: int = 1000,
    dir_path: str = None,
):
    logger = player_args.get("logger", logging.getLogger(__name__))
    
    BEST_MODEL_PATH = os.path.join(dir_path, "models/best.h5")
    ZERO_RESULTS_PATH = os.path.join(dir_path, "logs/zero_results.log")
    
    len_histories = []
    if not os.path.exists(ZERO_RESULTS_PATH):
        with open(ZERO_RESULTS_PATH, "w") as f:
            f.write("")
        
    with open(ZERO_RESULTS_PATH, "r") as f:
        for line in f.readlines():
            epoch, avg_len_history = line.split(",")
            epoch = int(epoch)
            avg_len_history = float(avg_len_history)
            len_histories += [(avg_len_history, epoch)]
    max_len_history = max(len_histories, key=lambda x: x[0]) if len(len_histories) else None

    adp_model = player(model_path=BEST_MODEL_PATH, **player_args)
    zero_model = None
    if zero_play:
        zero_model = AlphaZeroPlayer(**game_kwargs)
    
    buffer = deque(maxlen=buffer_size)
    scheduler = lr_scheduler.ExponentialLR(adp_model.nn.optimizer, gamma=lr_args['lr_decay'])
    
    for epoch in tqdm(range(epochs_start, epochs_end, epochs_step), position=0, leave=False, desc="Batches"):
        play_data = None
        if zero_play:
            play_data = [
                data[0] for data in 
                collect_play_data(
                    game_kwargs=game_kwargs, 
                    learner=adp_model,
                    trainer=zero_model,
                    n_games=epochs_step, 
                    epsilon=epsilon,
                )
            ]
        else:
            play_data = collect_selfplay_data(
                player=adp_model, 
                game_kwargs=game_kwargs, 
                n_games=epochs_step, 
                epsilon=epsilon,
            )
        
        buffer.extend(play_data)
        
        sample = random.sample(buffer, epochs_step)
        
        last_epoch_in_batch = epoch + epochs_step
        new_path = os.path.join(dir_path, "models/epoch_{}.h5".format(last_epoch_in_batch))
        
        if train:
            loss = adp_model.train_batch(sample, start=0, disable=False)
            lr = scheduler.get_last_lr()[-1]
            
            logger.info(f"Epoch: {epoch} to {epoch+epochs_step}, MSE: {loss}, LR: {lr:.5f}")
            
            scheduler.step()
            adp_model.nn.save_model(new_path)
            
        if eval:
            curr_model = player(model_path=new_path, **player_args)
            avg_len_history = eval_by_zero(
                game_kwargs=game_kwargs,
                curr_model=curr_model,
                zero_model=zero_model,
                n_test_games=n_test_games,
                epsilon=epsilon,
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
                    old_path = os.path.join(dir_path, "models/epoch_{}.h5".format(max_len_history[1]))
                    adp_model.nn.load_model(old_path)
                    logger.info("{} is loaded as the strongest model".format(old_path))
            else:
                adp_model.nn.save_model(BEST_MODEL_PATH)
                logger.info("{} is saved as the latest model".format(new_path))