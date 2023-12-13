from tqdm import tqdm
from .zero import AlphaZeroPlayer
from .adp import ADP_Player
from .mcts_adp import UCT_Zero_Player
from .players import Player
from .gomoku import Gomoku
from .data import collect_play_data, play_until_end

import torch
from collections import deque
from torch.optim import lr_scheduler
import os
import logging
import random
import math
import shutil

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

def eval_by_uct(game_kwargs: dict[str, int],
                curr_model: ADP_Player,
                best_model: ADP_Player,
                n_test_games: int,
                iterations: int = 10000,
                epsilon: float = .1) -> list[tuple[bool, bool]]:
    
    curr_uct_model = UCT_Zero_Player(adp_model=curr_model, iterations=iterations)
    best_uct_model = UCT_Zero_Player(adp_model=best_model, iterations=iterations)
    
    results = []
    for _ in tqdm(range(n_test_games), position=1, leave=False, desc="Testing"):
        win, curr_model_starts, _ = comp_models(game_kwargs, curr_uct_model, best_uct_model, epsilon1=epsilon)
        curr_model_won = int((win > 0) == curr_model_starts)
        results += [(curr_model_won, curr_model_starts)]
    
    return results

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
    def __init__(self, dir_path: str, logger, epoch: int = 0) -> tuple[int, float]:
        self.logger = logger
        self.dir_path = dir_path
        self.logger = logger
        self.latest_model_path = self.get_model_path(epoch)
        
        self.best_model_path = os.path.join(self.dir_path, "models/best.h5")
        if not os.path.exists(self.best_model_path):
            if os.path.exists(self.latest_model_path):
                shutil.copy(self.latest_model_path, self.best_model_path)
                logger.info("No best model, so use the latest model {} instead".format(self.latest_model_path))

        self.results_path = os.path.join(self.dir_path, "logs/mcts_results.log")
        
        self.max_win = None
        if not os.path.exists(self.results_path):
            with open(self.results_path, "w") as f:
                f.write("")
        else:
            with open(self.results_path, "r") as f:
                lines = f.readlines()
            lines = [line.strip().split(",") for line in lines]
            lines = [(int(line[0]), float(line[1])) for line in lines]
            if len(lines):
                self.max_win = max(lines, key=lambda x: x[1])
    
    def select_best(self, adp_model: ADP_Player, epoch: int, new_win_data: list[tuple[bool, bool]], select_best: bool = False):
        new_win_ratio = sum(map(lambda x: x[0], new_win_data)) / len(new_win_data)
        
        with open(self.results_path, "a") as f:
            f.write("{},{}\n".format((epoch, new_win_ratio)))
        
        path = self.get_model_path(epoch)
        if select_best:
            self.max_win = max(self.max_win, new_win_ratio, key=lambda x: x[1])
            if self.max_win == new_win_ratio:
                adp_model.nn.save_model(self.best_model_path)
                self.logger.info("{} is saved as the strongest model".format(path))
                
            else:
                old_path = self.get_model_path(self.max_win[0])
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
    eval_iterations: int = 2500,
    lr_args: dict = {},
    n_test_games: int = 0, 
    buffer_size: int = 1024,
    
    player_args: dict = {},
    epsilon: float = .25,
):
    logger = player_args.get("logger", logging.getLogger(__name__))
    
    evo_strategy = EvolutionStrategy(dir_path, logger, epochs_start)
    
    adp_model = player(
        model_path=evo_strategy.best_model_path, 
        game_kwargs=game_kwargs, 
        **player_args
    )
    
    zero_model = None
    if zero_play or eval:
        zero_model = AlphaZeroPlayer(game_kwargs)
    
    buffer = deque(maxlen=buffer_size)
    scheduler = lr_scheduler.ExponentialLR(adp_model.nn.optimizer, gamma=lr_args['lr_decay'])
    
    game = Gomoku(**game_kwargs)
    
    for epoch in tqdm(range(epochs_start, epochs_end, epochs_step), position=0, leave=False, desc="Epochs"):
        learner_args = {
            "player1": adp_model,
            "epsilon1": 1.,
        }
        
        trainer_args = {
            "player2": adp_model,
            "epsilon2": 0.
        }

        if zero_play:
            trainer_args = {
                "player2": zero_model,
                "epsilon2": .0,
            }
        
        buffer.extend(collect_play_data(
            game=game, 
            learner_args=learner_args,
            trainer_args=trainer_args,
            n_games=epochs_step,
        ))

        max_batch_size = min(epochs_step * 8, len(buffer))
        sample = random.sample(buffer, max_batch_size)
        
        last_epoch_in_batch = epoch + epochs_step
        
        if train:
            n_batches = int(math.ceil(max_batch_size / batch_size)) 
            pbar = tqdm(range(n_batches), position=1, leave=False, desc="Batches")
            for i in pbar:
                batch = sample[i::n_batches]
                mean_reward = sum([y for x, y in batch]) / len(batch)
                loss = adp_model.train_batch(sample, start=0)
                lr = scheduler.get_last_lr()[-1]
                
                completed_ratio = (i + 1) / n_batches
                pbar.set_postfix({"MSE": "{:.5f}".format(loss), "LR": "{:.5f}".format(lr)})
                
                logger.info(f"Epoch: {epoch} to {epoch+epochs_step}, Batch: {completed_ratio * 100 :.2f}%, Mean Reward: {mean_reward:.2f}, MSE: {loss:.5f}, LR: {lr:.5f}")
            
                scheduler.step()
            
            pbar.close()
            new_path = evo_strategy.get_model_path(last_epoch_in_batch)
            adp_model.nn.save_model(new_path)
            
        if eval:
            curr_model = player(
                model_path=new_path, 
                game_kwargs=game_kwargs,
                **player_args
            )
            
            win_data = eval_by_uct(
                game_kwargs=game_kwargs,
                curr_model=curr_model,
                best_model=adp_model,
                n_test_games=n_test_games,
                iterations=eval_iterations,
                epsilon=epsilon,
            )
            
            evo_strategy.select_best(
                adp_model=adp_model, 
                epoch=last_epoch_in_batch, 
                new_win_data=win_data, 
                select_best=select_best,
            )