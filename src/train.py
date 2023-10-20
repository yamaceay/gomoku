import random
import numpy as np
from collections import deque
import torch

from .mcts import Deep_Player
from .net import Zero_Net
from skrl.resources.schedulers.torch import KLAdaptiveLR
from .calc import kl_divergence, explained_var
from .data import play_n_games_for_train, extend_play_data, play_game
from .gomoku import Gomoku, S_GAME, M_GAME, L_GAME
from tqdm import tqdm
import os
import pickle 

TRAIN_ARGS = {
    "6_6_4": dict(n_zero = 400, n_uct = 5000, n_uct_step = 1000, n_uct_max = 5000),
    "8_8_5": dict(n_zero = 500, n_uct = 4500, n_uct_step = 1500, n_uct_max = 6000),
    "10_10_5": dict(n_zero = 600, n_uct = 6000, n_uct_step = 2000, n_uct_max = 6000),
}
class Trainer():
    def __init__(self,
                 game_kwargs: tuple[int, int, int],
                 train_kwargs: dict[str, int],
                 model_file: str = None,
                 
                 n_epochs: int = 1000,
                 batch_size: int = 512,
                 r_checkpoint: int = 50,
                 n_eval_games: int = 10,
                 n_batch_reps: int = 5,
                 buffer_size: int = 10000,
                 
                 lr: float = .000117,
                 weight_decay: float = .0001,
                 kl_threshold: float = .004,
                 
                 epsilon: float = .25,
                 temp: float = .0001,
                 k_ucb: float = 5,
                 gamma: float = .0,
                 ):
    
        self.game_kwargs = game_kwargs
        self.model_file = model_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_zero = train_kwargs["n_zero"]
        self.n_uct = train_kwargs["n_uct"]
        self.n_uct_step = train_kwargs["n_uct_step"]
        self.n_uct_max = train_kwargs["n_uct_max"]
        
        self.temp_max = 1.0
        self.temp = temp
        self.epsilon = epsilon
        self.gamma = gamma
        self.k_ucb = k_ucb
        self.next_state = gamma > 0
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_batch_reps = n_batch_reps
        self.n_eval_games = n_eval_games
        self.r_checkpoint = r_checkpoint
        self.buffer_size = buffer_size
        self.cache = deque(maxlen=self.buffer_size)
        
        self.lr_init = lr
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.n_min_games = 2
        self.best_win_ratio = 0.0
        self.perfect_win_ratio = 1.0

        self.net = Zero_Net(game_kwargs=self.game_kwargs,
                            model_file=self.model_file,
                            device=self.device,
                            opt_args=dict(lr=self.lr, weight_decay=self.weight_decay))
        
        self.scheduler = KLAdaptiveLR(self.net.optimizer, kl_threshold=kl_threshold)

    def fit(self) -> tuple[float, float, float, float, float]:
        mini_batch = random.sample(self.cache, self.batch_size)
        states, policies, rewards, *next_states = map(list, zip(*mini_batch))
        old_policy, old_value = self.net.forward_batch(states)
        
        kl_mean = .0
        for _ in range(self.n_batch_reps):
            batch = (states, policies, rewards, *next_states)
            loss, entropy = self.net.fit_one(batch, self.gamma)
            new_policy, new_value = self.net.forward_batch(states)
            kl_mean += kl_divergence(old_policy, new_policy)
        kl_mean /= self.n_batch_reps
        
        rewards = np.array(rewards)

        expl_var_prev = explained_var(rewards, old_value.reshape(-1))
        expl_var = explained_var(rewards, new_value.reshape(-1))
        
        self.scheduler.step(kl_mean)
        self.lr = self.scheduler.get_last_lr()[0]
        
        expl_var_diff = expl_var - expl_var_prev
            
        return loss, entropy, kl_mean, expl_var, expl_var_diff

    def test(self) -> tuple[float, list[int], float]:
        zero = Deep_Player(
            self.n_zero,
            policy_value_fn=self.net.predict,
            k_ucb=self.k_ucb,
            temp=self.temp,
        )
        uct = Deep_Player(
            self.n_uct,
            k_ucb=self.k_ucb,
            temp=self.temp,
        )
        
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
        batch_wins, batch_loses = self.buffer_size, self.buffer_size
        
        try:
            with open(BUFFER_PATH, "rb") as f:
                self.cache = deque(pickle.load(f), maxlen=self.buffer_size)
        except FileNotFoundError:
            batch_wins, batch_loses = 0, 0
            pass
        
        try:
            pbar = tqdm(range(self.n_epochs), position=0, leave=False, desc="Batches")
            for i in pbar:
                zero = Deep_Player(
                    self.n_zero,
                    policy_value_fn=self.net.predict,
                    k_ucb=self.k_ucb,
                    temp=self.temp_max,
                )
                
                game = Gomoku(*self.game_kwargs)
                
                play_data = play_n_games_for_train(game=game,
                                   player=zero, 
                                   epsilon=self.epsilon, 
                                   next_state=self.next_state)
                
                batch_wins += int(play_data[0][2] == 1)
                batch_loses += int(play_data[0][2] == -1)
                is_skewed = min(batch_wins, batch_loses) < self.n_min_games
                
                play_data = extend_play_data(play_data)
                episode_len = len(play_data) // 8
                self.cache.extend(play_data)
                
                logger.info(f"batch: {i+1}, len_episode: {episode_len}")
                if len(self.cache) > self.batch_size and not is_skewed:
                    loss, entropy, kl, expl_var, d_expl_var = self.fit()
                    fit_results = {
                        "kl": f"{kl:.5f}",
                        "lr": f"{self.lr:.6f}",
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
    import logging
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_size", type=str, choices=["S", "M", "L"])
    args = parser.parse_args()
    
    game_kwargs = S_GAME if args.game_size == "S" else M_GAME if args.game_size == "M" else L_GAME
    game_kwargs_str = "_".join(map(str, game_kwargs))

    train_kwargs = TRAIN_ARGS[game_kwargs_str]

    # game_kwargs_str += "_01"

    LOSSES_PATH = os.path.join(game_kwargs_str, f"train.log")
    BUFFER_PATH = os.path.join(game_kwargs_str, f"buf.pkl")
    CURR_MODEL_PATH = os.path.join(game_kwargs_str, f"models/curr.pkl")
    BEST_MODEL_PATH = os.path.join(game_kwargs_str, f"models/best.pkl")
    
    model_file = CURR_MODEL_PATH if os.path.exists(CURR_MODEL_PATH) else None

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    file_handler = logging.FileHandler(LOSSES_PATH)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    trainer = Trainer(
        game_kwargs=game_kwargs, 
        train_kwargs=train_kwargs,
        model_file=model_file
    )
    trainer.train()

