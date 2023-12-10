from .game import Board
from .mcts_alphaZero import MCTSPlayer as ZeroPlayer
from .players import Player
from .gomoku import Gomoku
from .policy_value_net_numpy import PolicyValueNetNumpy
import pickle
import os

ZERO_DIR_PATH = '_azero/models'

class AlphaZeroConv:
    def __init__(self, M: int, N: int, K: int):
        self.M = M
        self.N = N
        self.K = K
        
        model_file = os.path.join(ZERO_DIR_PATH, f'best_{self.M}_{self.N}_{self.K}')
        try:
            net_params = pickle.load(open(model_file, 'rb'))
        except:
            net_params = pickle.load(open(model_file, 'rb'),
                                        encoding='bytes')
        
        self.pre_nn = PolicyValueNetNumpy(
            board_height=self.M,
            board_width=self.N,
            net_params=net_params,
        )
    
    def forward(self, state: Gomoku):
        return self.pre_nn.policy_value_fn_frozen_conv(state)

class AlphaZeroPlayer(Player):
    def __init__(self, game_kwargs: dict[int], **kwargs):
        self.M = game_kwargs['M']
        self.N = game_kwargs['N']
        self.K = game_kwargs['K']
        
        c_puct = kwargs.get('c_puct', 5)
        n_playout = kwargs.get('n_playout', 1000)
        
        model_file = f'best_{self.M}_{self.N}_{self.K}'
        model_file = os.path.join(ZERO_DIR_PATH, model_file)
        
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                        encoding='bytes')  # To support python3
            
        self.best_policy = PolicyValueNetNumpy(self.M, self.N, policy_param)
        self.player = ZeroPlayer(
            self.best_policy.policy_value_fn,
            c_puct=c_puct,
            n_playout=n_playout,
        )
        
        self.board = Board(**game_kwargs)
        self.start_player = 0
        self.restart()
    
    def next_move_probs(self, game: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        n_moves = len(self.board.states)
        history = game.history()
        for location in history[n_moves:]:
            move = self.board.location_to_move(location)
            self.board.move(move)
            
        move = self.player.next_move(self.board)
        [x, y] = self.board.move_to_location(move)
        new_move = (int(x), int(y))
        return [(1., new_move)]
     
    def next_move(self, game: Gomoku, _: bool = True) -> tuple[int, int]:
        prob_action = self.next_move_probs(game)
        return prob_action[0][1]
    
    def restart(self):
        self.board.init_board(start_player=self.start_player)
        
# class PolicyValueNetTorch(PolicyValueNet):
#     """policy-value network """
#     def __init__(self, M: int, N: int, K: int, model_file: str = None):
#         self.board_width = M
#         self.board_height = N
#         self.n_in_rows = K
#         self.l2_const = 1e-4  # coef of l2 penalty
#         # the policy value net module
#         self.policy_value_net = Net(M, N).to(device)
#         self.optimizer = torch.optim.Adam(self.policy_value_net.parameters(),
#                                     weight_decay=self.l2_const)
#         self.use_gpu = torch.cuda.is_available()
#         self.optimizer = torch.optim.Adam(self.policy_value_net.parameters(),
#                                     weight_decay=self.l2_const)
#         if model_file:
#             try:
#                 policy_param = pickle.load(open(model_file, 'rb'))
#             except:
#                 policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')  # To support python3
                
#             net_params = {}
#             for v, (k, v_old) in zip(policy_param, self.policy_value_net.named_parameters()):
#                 expected = v_old.to(device)
#                 output = torch.as_tensor(v).to(device)
#                 if expected.size() != output.size():
#                     output = output.view(expected.size())
#                 assert expected.size() == output.size()
#                 net_params[k] = output

#             self.policy_value_net.load_state_dict(net_params)

# class TorchZeroPlayer(Player):
#     def __init__(self, M: int, N: int, K: int, **kwargs):
#         c_puct = kwargs.get('c_puct', 5)
#         n_playout = kwargs.get('n_playout', 1000)
        
#         model_file = f'best_{M}_{N}_{K}'
#         model_file = os.path.join(ZERO_DIR_PATH, model_file)
        
#         self.best_policy = PolicyValueNetTorch(M, N, K, model_file)
        
#         self.player = ZeroPlayer(
#             self.best_policy.policy_value_fn,
#             c_puct=c_puct,
#             n_playout=n_playout,
#         )
        
#         self.board = Board(M=M, N=N, K=K)
#         self.start_player = 0
#         self.restart()
    
#     def next_move_probs(self, _: Gomoku) -> list[tuple[float, tuple[int, int]]]:
#         raise NotImplementedError
     
#     def next_move(self, game: Gomoku) -> tuple[int, int]:
#         n_moves = len(self.board.states)
#         history = game.get_history()
#         for location in history[n_moves:]:
#             move = self.board.location_to_move(location)
#             self.board.move(move)
            
#         move = self.player.next_move(self.board)
#         [x, y] = self.board.move_to_location(move)
#         new_move = (int(x), int(y))
#         return new_move
    
#     def restart(self):
#         self.board.init_board(start_player=self.start_player)
