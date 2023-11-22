from .game import Game, Board
from .mcts_pure import MCTSPlayer as MCTS_Pure
from .mcts_alphaZero import MCTSPlayer
from .train import TrainPipeline
from .human_play import Human
from .policy_value_net_numpy import PolicyValueNetNumpy as PolicyValueNet
import pickle
import os

# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

if __name__ == "__main__":
    DIR = './models_azero'
    model_file = os.path.join(DIR, 'best_policy_8_8_5.model')
    try:
        policy_param = pickle.load(open(model_file, 'rb'))
    except:
        policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        
    best_policy = PolicyValueNet(8, 8, policy_param)
        
    game = Game(Board(width=8, height=8, n_in_row=5))
    mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)
    zero_player = MCTSPlayer(
        policy_value_function=best_policy.policy_value_fn, 
        c_puct=5, n_playout=1000
    )
    human = Human()
    
    game.start_play(human, zero_player, is_shown=1)