from typing import Callable
import numpy as np

from .net import Policy_Value_Net
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
    
if __name__ == "__main__":    
    game_kwargs = (6, 6, 4)

    model_file = f"bin/models/curr_{game_kwargs[0]}_{game_kwargs[1]}_{game_kwargs[2]}.model"
    net = Policy_Value_Net(
        game_kwargs=game_kwargs, 
        model_file=model_file,
    )

    list_iterations = [625, 1250, 2500, 5000]
    iterations = list_iterations[0]

    player1 = UCT_Player(
        k_ucb = 5,
        iterations = iterations,
        temp = .001,
    )

    player2 = UCT_No_Memory_Player(
        k_ucb = 5,
        iterations = iterations,
        temp = .001,
    )
    
    game = Gomoku(*game_kwargs)
    
    first_starts = True
    if not first_starts:
        action = player2.next_move(game)
        game.play(action)
        print(game)
    
    while not game.fin():
        action = player1.next_move(game)
        
        game.play(action)
        print(game)
        if game.fin():
            break
        action = player2.next_move(game)
        
        game.play(action)
        print(game)