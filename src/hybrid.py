from typing import Callable
import numpy as np

from .net import Zero_Net
from .gomoku import Gomoku
from .player import Player
from .calc import softmax
from .mcts import Deep_Player, uniform_probs

class Zero_Player(Deep_Player):
    def __init__(self, *args, **kwargs):
        kwargs["memory"] = True
        assert kwargs["policy_value_fn"] is not None, "policy_value_fn has to be assigned"
        super().__init__(*args, **kwargs)

class Zero_No_Policy_Player(Deep_Player):
    def __init__(self, *args, **kwargs):
        kwargs["memory"] = True
        policy_value_fn = kwargs["policy_value_fn"]
        assert policy_value_fn is not None, "policy_value_fn has to be assigned"
        kwargs["policy_value_fn"] = lambda s: (uniform_probs(s), policy_value_fn(s)[1])
        super().__init__(*args, **kwargs)
        
class UCT_Player(Deep_Player):
    def __init__(self, *args, **kwargs):
        kwargs["memory"] = True
        if kwargs.pop("policy_value_fn", None) is not None:
            print("policy_value_fn will be ignored") 
            
        super().__init__(*args, **kwargs)

class Zero_No_Memory_Player(Deep_Player):
    def __init__(self, *args, **kwargs):
        kwargs["memory"] = False
        assert kwargs["policy_value_fn"] is not None, "policy_value_fn has to be assigned"
        super().__init__(*args, **kwargs)

class Zero_No_Policy_No_Memory_Player(Deep_Player):
    def __init__(self, *args, **kwargs):
        kwargs["memory"] = False
        policy_value_fn = kwargs["policy_value_fn"]
        assert policy_value_fn is not None, "policy_value_fn has to be assigned"
        kwargs["policy_value_fn"] = lambda s: (uniform_probs(s), policy_value_fn(s)[1])
        super().__init__(*args, **kwargs)
  
class UCT_No_Memory_Player(Deep_Player):
    def __init__(self, *args, **kwargs):
        kwargs["memory"] = False
        if kwargs.pop("policy_value_fn", None) is not None:
            print("policy_value_fn will be ignored") 
            
        super().__init__(*args, **kwargs)

class Flat_Player(Player):
    def __init__(self, 
                 policy_value_fn: Callable,
                 temp: float = .001):
        
        self.policy_value_fn = policy_value_fn
        self.temp = temp
        
    def next_move_probs(self, state: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        probs_actions, _ = self.policy_value_fn(state) 
        probs, actions = zip(*probs_actions)
        probs = np.array(probs) 
            
        probs = np.log(probs + 1e-10)
        probs = softmax(probs / self.temp)
        
        return zip(probs, actions)