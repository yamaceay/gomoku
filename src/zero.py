from .game import Board
from .mcts_alphaZero import MCTSPlayer as ZeroPlayer
from .players import Player
from .gomoku import Gomoku, loc_to_move
from .policy_value_net_numpy import PolicyValueNetNumpy as PolicyValueNet
import pickle
import os

DIR = './models_azero'

class AlphaZeroPlayer(Player):
    def __init__(self, M: int, N: int, K: int, FIRST_PLAYER: int = 1, **kwargs):
        c_puct = kwargs.get('c_puct', 5)
        n_playout = kwargs.get('n_playout', 1000)
        
        model_file = f'best_{M}_{N}_{K}'
        model_file = os.path.join(DIR, model_file)
        
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                        encoding='bytes')  # To support python3
            
        self.best_policy = PolicyValueNet(M, N, policy_param)
        self.player = ZeroPlayer(
            self.best_policy.policy_value_fn,
            c_puct=c_puct,
            n_playout=n_playout,
        )
        
        self.board = Board(width=M, height=N, n_in_row=K)
        self.start_player = int(FIRST_PLAYER != 1)
        self.restart()
        
    def next_move(self, game: Gomoku) -> tuple[int, int]:
        n_moves = len(self.board.states)
        history = game.get_history()
        for location in history[n_moves:]:
            move = self.board.location_to_move(location)
            self.board.move(move)
            
        move = self.player.next_move(self.board)
        [x, y] = self.board.move_to_location(move)
        new_move = (int(x), int(y))
        return new_move
    
    def restart(self):
        self.board.init_board(start_player=self.start_player)