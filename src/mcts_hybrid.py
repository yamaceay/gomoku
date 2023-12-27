import numpy as np
from .players import Player
from .gomoku import Gomoku
from .mcts import UCT_Player
from .patterns import sortfn
from .adp import ADP_Player

class UCT_Tang_Player(Player):
    def __init__(self, 
                 adp_model: ADP_Player,
                 uct_model: UCT_Player,
                 k: int = 5,
                 C_ADP: int = 1):
        self.adp = adp_model
        self.uct = uct_model
        self.k = k
        self.C_ADP = C_ADP
        
    def next_move_probs(self, game: Gomoku):
        adp_probs, actions = zip(*self.adp.next_move_probs(game))
        actions = actions[:self.k]
        
        mcts_probs = []
        for action in actions:
            game_copy = game.copy()
            game_copy.play(action)
            uct = self.uct.copy()
            uct.next_move_probs(game_copy)
            mcts_probs += [uct.tree.root.n]
        
        mcts_probs = np.array(mcts_probs)
        mcts_probs = np.exp(mcts_probs - np.max(mcts_probs))
        mcts_probs /= np.sum(mcts_probs)
        
        total_probs = [
            (mcts_probs[i] + self.C_ADP * adp_probs[i], actions[i])
            for i in range(self.k)
        ]
        
        return sortfn(total_probs)