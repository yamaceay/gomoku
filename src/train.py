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

M, N, K = 10, 10, 5
game_kwargs = (M, N, K)

DIR = '_zero'
LOSSES_PATH = os.path.join(DIR, "logs/losses5.log")
MODELS_PATH = os.path.join(DIR, "models")
BUFFER_PATH = os.path.join(DIR, "data_buffer5.pkl")
CURR_MODEL_PATH = os.path.join(MODELS_PATH, f"curr_{M}_{N}_{K}.model1")
BEST_MODEL_PATH = os.path.join(MODELS_PATH, f"best_{M}_{N}_{K}.model1")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
file_handler = logging.FileHandler(LOSSES_PATH)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class TrainPipeline():
    def __init__(self,
                 model_file: str = None,
                 
                 n_zero: int = 800,
                 n_uct: int = 2000,
                 n_uct_step: int = 2000,
                 n_uct_max: int = 10000,
                 
                 n_batches: int = 1500,
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
                 c_puct: float = 5,
                 next_state: bool = False,
                 gamma: float = .9,
                 ):
        # params of the board and the game
        self.game_kwargs = game_kwargs
        # training params
        self.lr = lr
        self.lr_multiplier = lr_multiplier  # adaptively adjust the learning rate based on KL
        self.lr_step = lr_step
        self.lr_range = lr_range
        self.temp = temp  # the temperature param
        self.epsilon = epsilon # the epsilon greedy param for self-play policy
        self.next_state = next_state
        self.gamma = gamma # discount factor for next state
        self.n_zero = n_zero  # num of simulations for each move
        self.c_puct = c_puct
        self.buffer_size = buffer_size
        self.batch_size = batch_size  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = n_games_per_batch
        self.epochs = n_epochs  # num of train_steps for each update
        self.kl_targ = kl_targ
        self.kl_range = kl_range
        self.check_freq = r_checkpoint
        self.n_batches = n_batches
        self.best_win_ratio = 0.0
        self.perfect_win_ratio = 1.0
        self.model_file = model_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.n_uct = n_uct
        self.n_uct_step = n_uct_step
        self.n_uct_max = n_uct_max
        # start training from an initial policy-value net
        self.policy_value_net = Policy_Value_Net(game_kwargs=self.game_kwargs,
                                                 model_file=self.model_file,
                                                 device=self.device)

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        current_mcts_player = Deep_Player(policy_value_fn=self.policy_value_net.policy_value_fn_sorted,
                                          c_puct=self.c_puct,
                                          iterations=self.n_zero,
                                          temp=self.temp)
        
        game = Gomoku(*self.game_kwargs)
        for i in range(n_games):
            play_data = play_n_games_for_train(game, 1, current_mcts_player, self.epsilon, self.next_state)
            play_data = extend_play_data(play_data)
            self.episode_len = len(play_data) // 8
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch, mcts_probs_batch, winner_batch, *next_state_batch = map(list, zip(*mini_batch))
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        next_old_v = np.zeros_like(old_v)
        if self.next_state:
            _, next_old_v = self.policy_value_net.policy_value(next_state_batch[0])
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    (state_batch, mcts_probs_batch, winner_batch, *next_state_batch),
                    self.lr*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = kl_divergence(old_probs, new_probs)
            if kl > 2 * self.kl_targ * self.kl_range:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * self.kl_range and self.lr_multiplier * self.lr_range > 1:
            self.lr_multiplier /= self.lr_step
        elif kl < self.kl_targ / self.kl_range and self.lr_multiplier / self.lr_range < 1:
            self.lr_multiplier *= self.lr_step
        
        winner_batch = np.array(winner_batch)

        next_new_v = np.zeros_like(new_v)
        if self.next_state:
            _, next_new_v = self.policy_value_net.policy_value(next_state_batch[0])

        explained_var_old = explained_var(winner_batch + next_old_v.flatten(), old_v.flatten())
        explained_var_new = explained_var(winner_batch + next_new_v.flatten(), new_v.flatten())
            
        return loss, entropy, kl, explained_var_old, explained_var_new

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = Deep_Player(policy_value_fn=self.policy_value_net.policy_value_fn_sorted,
                                         c_puct=self.c_puct,
                                         iterations=self.n_zero,
                                         temp=self.temp)
        pure_mcts_player = Deep_Player(c_puct=self.c_puct,
                                     iterations=self.n_uct,
                                     temp=self.temp)
        win_cnt = defaultdict(int)
        game = Gomoku(*self.game_kwargs)
        avg_curr_starts = .0
        for i in range(n_games):
            end_game, curr_starts, _ = play_game(
                game, 
                current_mcts_player, 
                pure_mcts_player)
            winner = end_game.score()
            if not curr_starts:
                winner = -winner
            win_cnt[winner] += 1
            avg_curr_starts += curr_starts
        win_ratio = (1.0*win_cnt[1] + 0.5*win_cnt[0]) / n_games
        avg_curr_starts /= n_games
        return win_ratio, win_cnt, avg_curr_starts

    def train(self):
        """run the training pipeline"""
        # read buffer if possible
        try:
            with open(BUFFER_PATH, "rb") as f:
                self.data_buffer = deque(pickle.load(f), maxlen=self.buffer_size)
        except FileNotFoundError:
            pass
        
        try:
            pbar = tqdm(range(self.n_batches), position=0, leave=False, desc="Batches")
            for i in pbar:
                self.collect_selfplay_data(self.play_batch_size)
                logger.info(f"batch: {i+1}, len_episode: {self.episode_len}")
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy, kl, explained_var_old, explained_var_new = self.policy_update()
                    policy_update_results = {
                        "kl": f"{kl:.5f}",
                        "lr": f"{self.lr * self.lr_multiplier:.3f}",
                        "loss": f"{loss:.5f}",
                        "entropy": f"{entropy:.5f}",
                        "expl_var_diff": f"{explained_var_new - explained_var_old:.3f}",
                        "expl_var": f"{explained_var_new:.3f}",
                    }
                    pbar.set_postfix(policy_update_results)
                    logger.info(", ".join([f"{k}: {v}" for k, v in policy_update_results.items()]))
                
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    logger.info(f"evalBatch: {i+1}")
                    win_ratio, win_cnt, avg_curr_starts = self.policy_evaluate()
                    self.policy_value_net.save_model(CURR_MODEL_PATH)
                    if win_ratio > self.best_win_ratio:
                        logger.info("BEST POLICY!!!!!!!")
                    
                    policy_evaluate_results = {
                        "n_uct": f"{self.n_uct}",
                        "win": f"{win_cnt[1]}",
                        "lose": f"{win_cnt[-1]}",
                        "tie": f"{win_cnt[0]}",
                        "first_turn_rate": f"{avg_curr_starts:.3f}",
                    }
                    logger.info(", ".join([f"{k}: {v}" for k, v in policy_evaluate_results.items()]))
                    
                    if win_ratio > self.best_win_ratio:
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(BEST_MODEL_PATH)
                        
                    if win_ratio >= self.perfect_win_ratio and self.n_uct < self.n_uct_max:
                        self.n_uct += self.n_uct_step
                        self.best_win_ratio = 0.0
                        
                    elif win_ratio <= 1 - self.perfect_win_ratio and self.n_uct > self.n_uct_step:
                        self.n_uct -= self.n_uct_step
                        self.best_win_ratio = 0.0
                            
        except KeyboardInterrupt:
            print('\n\rquit')

        # save buffer
        with open(BUFFER_PATH, "wb") as f:
            pickle.dump(list(self.data_buffer), f)

if __name__ == '__main__':
    # training_pipeline = TrainPipeline(init_model=CURR_MODEL_PATH)
    training_pipeline = TrainPipeline()
    training_pipeline.train()

