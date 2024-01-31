import random
import numpy as np

from collections import defaultdict, deque
from .mcts import Deep_Player
from .net import Policy_Value_Net
from .calc import kl_divergence, explained_var
from .data import play_n_games_for_train, extend_play_data, play_game
from .gomoku import Gomoku
import torch
from tqdm import tqdm
import os
import logging
import pickle 

game_kwargs = (M, N, K) = 6, 6, 4
game_kwargs_str = f"{M}_{N}_{K}"

DIR = 'bin'
LOSSES_PATH = os.path.join(DIR, f"logs/{game_kwargs_str}.log")
BUFFER_PATH = os.path.join(DIR, f"buf_{game_kwargs_str}.pkl")
CURR_MODEL_PATH = os.path.join(DIR, f"models/curr_{game_kwargs_str}.model")
BEST_MODEL_PATH = os.path.join(DIR, f"models/best_{game_kwargs_str}.model")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
file_handler = logging.FileHandler(LOSSES_PATH)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class TrainPipeline():
    def __init__(self,
                 model_file: str = None,
                 
                 n_zero: int = 400, # S: 400, M: 500, L: 600
                 n_uct: int = 1000, # S: 1000, M: 1500, L: 2000
                 n_uct_step: int = 1000, # S: 1000, M: 1500, L: 2000
                 n_uct_max: int = 5000, # S: 5000, M: 7500, L: 10000
                 
                 n_batches: int = 1500, # S: 1500, M: 1500, L: 1000
                 batch_size: int = 512,
                 n_games_per_batch: int = 1,
                 r_checkpoint: int = 50,
                 n_epochs: int = 5,
                 buffer_size: int = 10000,
                 
                 lr: float = .002,
                 lr_multiplier: float = 1,
                 lr_step: float = 1.5,
                 lr_range: float = 5,
                 
                 kl_targ: float = 0.02,
                 kl_range: float = 2,
                 
                 epsilon: float = .25,
                 temp: float = .001,
                 k_ucb: float = 5,
                 gamma: float = .9,
                 ):
    
        self.game_kwargs = game_kwargs
        self.model_file = model_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_zero = n_zero
        self.n_uct = n_uct
        self.n_uct_step = n_uct_step
        self.n_uct_max = n_uct_max
        
        self.temp = temp
        self.epsilon = epsilon
        self.gamma = gamma 
        self.k_ucb = k_ucb
        self.next_state = gamma > 0
        
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_games_per_batch = n_games_per_batch
        self.n_epochs = n_epochs
        self.r_checkpoint = r_checkpoint
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=self.buffer_size)
        
        self.lr = lr
        self.lr_multiplier = lr_multiplier
        self.lr_step = lr_step
        self.lr_range = lr_range
        
        self.kl_targ = kl_targ
        self.kl_range = kl_range
        
        self.best_win_ratio = 0.0
        self.perfect_win_ratio = 1.0

        self.net = Policy_Value_Net(game_kwargs=self.game_kwargs,
                                    model_file=self.model_file,
                                    device=self.device)

    def collect_selfplay_data(self, n_games=1):
        zero = Deep_Player(policy_value_fn=self.net.policy_value_fn_sorted,
                                          iterations=self.n_zero,
                                          k_ucb=self.k_ucb,
                                          temp=self.temp)
        
        game = Gomoku(*self.game_kwargs)
        for i in range(n_games):
            play_data = play_n_games_for_train(game, 1, zero, self.epsilon, self.next_state)
            play_data = extend_play_data(play_data)
            self.episode_len = len(play_data) // 8
            self.data_buffer.extend(play_data)

    def fit(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch, mcts_probs_batch, winner_batch, *next_state_batch = map(list, zip(*mini_batch))
        old_probs, old_v = self.net.policy_value(state_batch)
        for _ in range(self.n_epochs):
            loss, entropy = self.net.fit_one(
                    (state_batch, mcts_probs_batch, winner_batch, *next_state_batch),
                    self.lr*self.lr_multiplier, self.gamma)
            new_probs, new_v = self.net.policy_value(state_batch)
            kl = kl_divergence(old_probs, new_probs)
            if kl > 2 * self.kl_targ * self.kl_range:
                break

        if kl > self.kl_targ * self.kl_range and self.lr_multiplier * self.lr_range > 1:
            self.lr_multiplier /= self.lr_step
        elif kl < self.kl_targ / self.kl_range and self.lr_multiplier / self.lr_range < 1:
            self.lr_multiplier *= self.lr_step
        
        winner_batch = np.array(winner_batch)

        expl_var_prev = explained_var(winner_batch, old_v.flatten())
        expl_var = explained_var(winner_batch, new_v.flatten())
        
        expl_var_diff = expl_var - expl_var_prev
            
        return loss, entropy, kl, expl_var, expl_var_diff

    def test(self, n_games=10):
        zero = Deep_Player(policy_value_fn=self.net.policy_value_fn_sorted,
                           k_ucb=self.k_ucb,
                           iterations=self.n_zero,
                           temp=self.temp)
        uct = Deep_Player(k_ucb=self.k_ucb,
                          iterations=self.n_uct,
                          temp=self.temp)
        outcomes = [0, 0, 0] # tie, win, lose
        game = Gomoku(*self.game_kwargs)
        avg_curr_starts = .0
        for i in range(n_games):
            end_game, curr_starts, _ = play_game(game, zero, uct)
            winner = end_game.score()
            if not curr_starts:
                winner = -winner
            outcomes[winner] += 1
            avg_curr_starts += curr_starts
        win_ratio = outcomes[1] + outcomes[0] / 2 
        win_ratio /= n_games
        avg_curr_starts /= n_games
        return win_ratio, outcomes, avg_curr_starts

    def train(self):
        try:
            with open(BUFFER_PATH, "rb") as f:
                self.data_buffer = deque(pickle.load(f), maxlen=self.buffer_size)
        except FileNotFoundError:
            pass
        
        try:
            pbar = tqdm(range(self.n_batches), position=0, leave=False, desc="Batches")
            for i in pbar:
                self.collect_selfplay_data(self.n_games_per_batch)
                logger.info(f"batch: {i+1}, len_episode: {self.episode_len}")
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy, kl, expl_var, d_expl_var = self.fit()
                    fit_results = {
                        "kl": f"{kl:.5f}",
                        "lr": f"{self.lr * self.lr_multiplier:.6f}",
                        "loss": f"{loss:.5f}",
                        "entropy": f"{entropy:.5f}",
                        "expl_var": f"{expl_var:.3f}",
                        "d_expl_var": f"{d_expl_var:.3f}",
                    }
                    pbar.set_postfix(fit_results)
                    logger.info(", ".join([f"{k}: {v}" for k, v in fit_results.items()]))
                
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.r_checkpoint == 0:
                    logger.info(f"eval_batch: {i+1}")
                    win_ratio, outcomes, avg_curr_starts = self.test()
                    self.net.save_model(CURR_MODEL_PATH)
                    if win_ratio > self.best_win_ratio:
                        logger.info("BEST POLICY!!!!!!!")
                    
                    test_results = {
                        "n_uct": f"{self.n_uct}",
                        "win": f"{outcomes[1]}",
                        "lose": f"{outcomes[-1]}",
                        "tie": f"{outcomes[0]}",
                        "first_turn_rate": f"{avg_curr_starts:.3f}",
                    }
                    logger.info(", ".join([f"{k}: {v}" for k, v in test_results.items()]))
                    
                    if win_ratio > self.best_win_ratio:
                        self.best_win_ratio = win_ratio
                        self.net.save_model(BEST_MODEL_PATH)
                        
                    if win_ratio >= self.perfect_win_ratio and self.n_uct < self.n_uct_max:
                        self.n_uct += self.n_uct_step
                        self.best_win_ratio = 0.0
                        
                    elif win_ratio <= 1 - self.perfect_win_ratio and self.n_uct > self.n_uct_step:
                        self.n_uct -= self.n_uct_step
                        self.best_win_ratio = 0.0
                            
        except KeyboardInterrupt:
            print('\n\rStopped')

        # save buffer
        with open(BUFFER_PATH, "wb") as f:
            pickle.dump(list(self.data_buffer), f)

if __name__ == '__main__':
    # training_pipeline = TrainPipeline(init_model=CURR_MODEL_PATH)
    training_pipeline = TrainPipeline()
    training_pipeline.train()

