import random
import numpy as np

from collections import deque
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
LOSSES_PATH = os.path.join(DIR, f"logs/{game_kwargs_str}_x.log")
BUFFER_PATH = os.path.join(DIR, f"buf_{game_kwargs_str}_x.pkl")
CURR_MODEL_PATH = os.path.join(DIR, f"models/curr_{game_kwargs_str}_x.model")
BEST_MODEL_PATH = os.path.join(DIR, f"models/best_{game_kwargs_str}_x.model")

TRAIN_ARGS = {
    "6_6_4": dict(n_zero = 400, n_uct = 1000, n_uct_step = 1000, n_uct_max = 5000, n_batches = 1500),
    "8_8_5": dict(n_zero = 500, n_uct = 1500, n_uct_step = 1500, n_uct_max = 6000, n_batches = 1500),
    "10_10_5": dict(n_zero = 600, n_uct = 2000, n_uct_step = 2000, n_uct_max = 6000, n_batches = 1000),
}

assert game_kwargs_str in TRAIN_ARGS, f"stringified game kwargs must be in {list(TRAIN_ARGS.keys())}"
train_kwargs = TRAIN_ARGS[game_kwargs_str]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
file_handler = logging.FileHandler(LOSSES_PATH)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class Trainer():
    def __init__(self,
                 model_file: str = None,
                 
                 n_zero: int = train_kwargs["n_zero"],
                 n_uct: int = train_kwargs["n_uct"],
                 n_uct_step: int = train_kwargs["n_uct_step"],
                 n_uct_max: int = train_kwargs["n_uct_max"],
                 
                 n_batches: int = train_kwargs["n_batches"],
                 batch_size: int = 512,
                 r_checkpoint: int = 50,
                 n_eval_games: int = 10,
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
                 gamma: float = .0,
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
        self.n_epochs = n_epochs
        self.n_eval_games = n_eval_games
        self.r_checkpoint = r_checkpoint
        self.buffer_size = buffer_size
        self.cache = deque(maxlen=self.buffer_size)
        
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

    def fit(self) -> tuple[float, float, float, float, float]:
        mini_batch = random.sample(self.cache, self.batch_size)
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

    def test(self) -> tuple[float, list[int], float]:
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
        for _ in range(self.n_eval_games):
            end_game, curr_starts, _ = play_game(game, zero, uct)
            winner = end_game.score()
            if not curr_starts:
                winner = -winner
            outcomes[winner] += 1
            avg_curr_starts += curr_starts
        win_ratio = outcomes[1] + outcomes[0] / 2 
        win_ratio /= self.n_eval_games
        avg_curr_starts /= self.n_eval_games
        return win_ratio, outcomes, avg_curr_starts

    def train(self):
        try:
            with open(BUFFER_PATH, "rb") as f:
                self.cache = deque(pickle.load(f), maxlen=self.buffer_size)
        except FileNotFoundError:
            pass
        
        try:
            pbar = tqdm(range(self.n_batches), position=0, leave=False, desc="Batches")
            for i in pbar:
                zero = Deep_Player(policy_value_fn=self.net.policy_value_fn_sorted,
                                   iterations=self.n_zero,
                                   k_ucb=self.k_ucb,
                                   temp=self.temp)
                game = Gomoku(*self.game_kwargs)
                play_data = play_n_games_for_train(game=game,
                                                   player=zero, 
                                                   epsilon=self.epsilon, 
                                                   next_state=self.next_state)
                
                play_data = extend_play_data(play_data)
                self.episode_len = len(play_data) // 8
                self.cache.extend(play_data)
                
                logger.info(f"batch: {i+1}, len_episode: {self.episode_len}")
                if len(self.cache) > self.batch_size:
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
                    pbar.set_postfix(test_results)
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

        with open(BUFFER_PATH, "wb") as f:
            pickle.dump(list(self.cache), f)

if __name__ == '__main__':
    # trainer = Trainer(init_model=CURR_MODEL_PATH)
    trainer = Trainer()
    trainer.train()

