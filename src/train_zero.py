from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from .game import Board, Game
from .mcts import UCT_Player
from .policy_value_net_pytorch import PolicyValueNet

class TrainPipeline():
    def __init__(self, 
                 M: int = 6,
                 N: int = 6,
                 K: int = 4,
                 init_model: str = None,
                 lr: float = 2e-3,
                 lr_multiplier: float = 1.0,
                 temp: float = .001,
                 epsilon: float = .25,
                 n_playout: int = 400,
                 c_puct: float = 5,
                 buffer_size: int = 10000,
                 batch_size: int = 512,
                 play_batch_size: int = 1,
                 epochs: int = 5,
                 kl_targ: float = 0.02,
                 check_freq: int = 50,
                 game_batch_num: int = 1500,
                 pure_mcts_playout_num: int = 1000,
                 ):
        # params of the board and the game
        self.M = M
        self.N = N
        self.K = K
        self.board = Board(M=self.M,
                           N=self.N,
                           K=self.K)
        self.game: Game = Game(self.board)
        # training params
        self.lr = lr
        self.lr_multiplier = lr_multiplier  # adaptively adjust the learning rate based on KL
        self.temp = temp  # the temperature param
        self.epsilon = epsilon # the epsilon greedy param for self-play policy
        self.n_playout = n_playout  # num of simulations for each move
        self.c_puct = c_puct
        self.buffer_size = buffer_size
        self.batch_size = batch_size  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = play_batch_size
        self.epochs = epochs  # num of train_steps for each update
        self.kl_targ = kl_targ
        self.check_freq = check_freq
        self.game_batch_num = game_batch_num
        self.best_win_ratio = 0.0
        self.model_file = init_model
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = pure_mcts_playout_num
        # start training from an initial policy-value net
        self.policy_value_net = PolicyValueNet(self.M,
                                                self.N,
                                                model_file=init_model)
        self.mcts_player = UCT_Player(policy_value_fn=self.policy_value_net.policy_value_fn,
                                      policy_kwargs={'C': 5},
                                      iterations=self.n_playout)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.N, self.M)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.epsilon)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.lr*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = UCT_Player(policy_value_fn=self.policy_value_net.policy_value_fn,
                                         policy_kwargs={'C': 5},
                                         iterations=self.n_playout)
        pure_mcts_player = UCT_Player(policy_kwargs={'C': 5},
                                     iterations=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline(init_model="./current_policy.model")
    training_pipeline.run()
