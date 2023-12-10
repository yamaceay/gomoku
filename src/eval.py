from tqdm import tqdm
from .zero import AlphaZeroPlayer
from .adp import ADP_Player
from .players import Player
from .gomoku import Gomoku
from .data import collect_play_data, play_until_end

import torch
from collections import deque
from torch.optim import lr_scheduler
import os
import logging
import random

def comp_models(game_kwargs: dict[str, int],
    player1: Player,
    player2: Player,
    epsilon1: float = .0,
    epsilon2: float = .0) -> list[tuple[str, bool]]:
        
    learner_args = {
        "player1": player1,
        "epsilon1": epsilon1,
    }
    
    trainer_args = {
        "player2": player2,
        "epsilon2": epsilon2,
    }
    
    game = Gomoku(**game_kwargs)
    
    with torch.no_grad():
        game, learner_starts = play_until_end(game, **learner_args, **trainer_args)
    
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

def eval_by_zero(game_kwargs: dict[str, int], 
                 curr_model: ADP_Player, 
                 zero_model: AlphaZeroPlayer, 
                 n_test_games: int, 
                 epsilon: float = .1):
    len_histories = []
    for _ in tqdm(range(n_test_games), position=1, leave=False, desc="Testing"):
        win, curr_model_starts, len_history = comp_models(game_kwargs, curr_model, zero_model, epsilon1=epsilon)
        zero_model.restart()
        curr_model_won = int((win > 0) == curr_model_starts)
        if curr_model_won:
            print("Current model won against zero, amazing!!")
            return game_kwargs['M'] * game_kwargs['N']
        len_histories += [len_history]
    avg_len_history = sum(len_histories) / len(len_histories)
    return avg_len_history
 
class EvolutionStrategy:
    def __init__(self, dir_path: str, logger) -> tuple[int, float]:
        self.dir_path = dir_path
        self.logger = logger
        
        self.best_model_path = os.path.join(self.dir_path, "models/best.h5")
        self.results_path = os.path.join(self.dir_path, "logs/zero_results.log")
        
        self.max_len_history = None
        if not os.path.exists(self.results_path):
            with open(self.results_path, "w") as f:
                f.write("")
        else:
            with open(self.results_path, "r") as f:
                lines = f.readlines()
            lines = [line.strip().split(",") for line in lines]
            lines = [(int(line[0]), float(line[1])) for line in lines]
            if len(lines):
                self.max_len_history = max(lines, key=lambda x: x[1])
    
    def select_best(self, adp_model: ADP_Player, path: str, new_len_history: tuple[int, float], select_best: bool = False):
        with open(self.results_path, "a") as f:
            f.write("{},{}\n".format(*new_len_history))
        
            if select_best:
                self.max_len_history = max(self.max_len_history, new_len_history, key=lambda x: x[1])
                if self.max_len_history == new_len_history:
                    adp_model.nn.save_model(self.best_model_path)
                    self.logger.info("{} is saved as the strongest model".format(path))
                    
                else:
                    old_path = self.get_model_path(self.max_len_history[0])
                    adp_model.nn.load_model(old_path)
                    self.logger.info("{} is loaded as the strongest model".format(old_path))
            else:
                adp_model.nn.save_model(self.best_model_path)
                self.logger.info("{} is saved as the latest model".format(path))
        return adp_model
      
    def get_model_path(self, epoch: int) -> str:
        return os.path.join(self.dir_path, "models/epoch_{}.h5".format(epoch))
      
def train_adp(
    dir_path: str,
    player: ADP_Player,
    
    epochs_end: int, 
    epochs_step: int,
    game_kwargs: dict[str, int], 
    
    batch_size: int,
    
    epochs_start: int = 0, 
    eval: bool = True, 
    train: bool = True,
    zero_play: bool = False,
    select_best: bool = False,
    lr_args: dict = {},
    n_test_games: int = 0, 
    buffer_size: int = 1024,
    
    player_args: dict = {},
    epsilon: float = .1,
):
    logger = player_args.get("logger", logging.getLogger(__name__))
    
    history_eval = EvolutionStrategy(dir_path, logger)
    
    adp_model = player(
        model_path=history_eval.best_model_path, 
        game_kwargs=game_kwargs, 
        **player_args
    )
    
    zero_model = None
    if zero_play or eval:
        zero_model = AlphaZeroPlayer(game_kwargs)
    
    buffer = deque(maxlen=buffer_size)
    scheduler = lr_scheduler.ExponentialLR(adp_model.nn.optimizer, gamma=lr_args['lr_decay'])
    
    game = Gomoku(**game_kwargs)
    
    for epoch in tqdm(range(epochs_start, epochs_end, epochs_step), position=0, leave=False, desc="Batches"):
        play_data = None
        
        learner_args = {
            "player1": adp_model,
            "epsilon1": epsilon,
        }
        
        trainer_args = {}
        if zero_play:
            trainer_args = {
                "player2": zero_model,
                "epsilon2": .0,
            }
            
        play_data = collect_play_data(
            game=game, 
            learner_args=learner_args,
            trainer_args=trainer_args,
            n_games=epochs_step,
        )
        
        buffer.extend(play_data)
        
        sample = random.sample(buffer, min(batch_size, len(buffer)))
        
        last_epoch_in_batch = epoch + epochs_step
        new_path = os.path.join(dir_path, "models/epoch_{}.h5".format(last_epoch_in_batch))
        
        if train:
            loss = adp_model.train_batch(sample, start=0)
            lr = scheduler.get_last_lr()[-1]
            
            logger.info(f"Epoch: {epoch} to {epoch+epochs_step}, MSE: {loss}, LR: {lr:.5f}")
            
            scheduler.step()
            adp_model.nn.save_model(new_path)
            
        if eval:
            curr_model = player(
                model_path=new_path, 
                game_kwargs=game_kwargs,
                **player_args
            )
            
            avg_len_history = eval_by_zero(
                game_kwargs=game_kwargs,
                curr_model=curr_model,
                zero_model=zero_model,
                n_test_games=n_test_games,
                epsilon=epsilon,
            )
            
            history_eval.select_best(
                adp_model=adp_model, 
                path=new_path, 
                new_len_history=(last_epoch_in_batch, avg_len_history), 
                select_best=select_best,
            )