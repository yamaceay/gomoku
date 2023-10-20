import copy
from .gomoku import Gomoku, sortfn
import numpy as np

class Player:
    def copy(self):
        return copy.deepcopy(self)
    
    def next_move_probs(self, game: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        raise NotImplementedError
    
    def next_move(self, 
                  state: Gomoku, 
                  epsilon: float = .0, 
                  get_probs: bool = False) -> tuple[int, int] | list[tuple[float, tuple[int, int]]]:
        
        probs_actions = self.next_move_probs(state)
        probs, actions = zip(*probs_actions)
        if epsilon != .0:
            probs += epsilon * (self.noise(len(probs)) - probs)

        probs /= sum(probs)
        
        action_i = np.random.choice(list(range(len(actions))), p=probs)
        action = actions[action_i]
        if get_probs:
            return action, zip(probs, actions)
        return action

class Rand_Player(Player):
    def next_move_probs(self, game: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        actions = game.actions()
        probs = np.random.random(len(actions))
        probs /= probs.sum()
        probs_actions = [(probs[i], actions[i]) for i in range(len(actions))]
        probs_actions = sortfn(probs_actions)
        return probs_actions