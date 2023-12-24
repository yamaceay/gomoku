from .gomoku import Gomoku
from .players import Player
from tqdm import tqdm
import random
from .mcts import UCT_Player
import numpy as np

def play_until_end(
    game: Gomoku, 
    player1: Player = None, 
    player2: Player = None,
    epsilon1: float = .0,
    epsilon2: float = .0,
    ) -> tuple[Gomoku, bool]:
    
    if game.fin() or (player1 is None and player2 is None):
        return game, True
        
    new_game = game.copy()
    
    if player2 is None:
        while not new_game.fin():
            action = player1.next_move(new_game, epsilon=epsilon1)
            new_game.play(action)
        return new_game, True    
    
    player2_starts = random.random() < .5
    if player2_starts:
        action = player2.next_move(new_game, epsilon=epsilon2)
        new_game.play(action)
    
    while not new_game.fin():
        action = player1.next_move(new_game, epsilon=epsilon1)
        new_game.play(action)
        if new_game.fin():
            break
        action = player2.next_move(new_game, epsilon=epsilon2)
        new_game.play(action)

    return new_game, not player2_starts

    
def collect_play_data(
    game: Gomoku,
    n_games: int = 1, 
    learner_args: dict[str] = {},
    trainer_args: dict[str] = {},
    ) -> list[tuple[str, float]]:
    
    round_kwargs = {}
    if "player" in learner_args:
        round_kwargs["player1"] = learner_args["player"]
    if "player" in trainer_args:
        round_kwargs["player2"] = trainer_args["player"]
    if "epsilon" in learner_args:
        round_kwargs["epsilon1"] = learner_args["epsilon"]
    if "epsilon" in trainer_args:
        round_kwargs["epsilon2"] = trainer_args["epsilon"]
        
    play_data = []
    for _ in tqdm(range(n_games), 
                  position=1, 
                  leave=False, 
                  desc="Collecting play data",
                  disable=True):
        
        new_game, _ = play_until_end(game, **round_kwargs)
        for feature in new_game.history_str_aug():
            label = new_game.score()
            play_data += [(feature, label)]
    
    return play_data

def play_self_until_end_zero(
    game: Gomoku, 
    player: UCT_Player = None, 
    epsilon: float = .0,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    
    if game.fin() or player is None:
        return []
        
    list_states = []
    list_probs = []
    new_game = game.copy()
    
    for _ in tqdm(range(game.M * game.N)):
        if new_game.fin():
            break
        list_states += [new_game.to_zero_input()]
        
        action, probs = player.next_move_data(
            new_game, 
            epsilon=epsilon,
        )
        
        list_probs += [probs]
        new_game.play(action)

    winner = new_game.score()
    list_winner = np.ones(len(list_probs)) * winner
    return zip(list_states, list_probs, list_winner)  


def collect_self_play_data_zero(
    game: Gomoku,
    n_games: int = 1, 
    player: UCT_Player = None,
    epsilon: float = .0,
    ) -> list[tuple[str, float]]:
    
    play_data = []
    for _ in tqdm(range(n_games), 
                  position=1, 
                  leave=False, 
                  desc="Collecting self-play data",
                  disable=True):
        for state, mcts_prob, winner in play_self_until_end_zero(game, player=player, epsilon=epsilon):
            play_data += [(state, mcts_prob, winner)]
    return play_data

def extend_play_data(
    play_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    
    extend_data = []
    for state, mcts_prob, winner in play_data:
        _, m, n = state.shape
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(
                mcts_prob.reshape(m, n)), i)
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

if __name__ == '__main__':
    from .policy_value_net import PolicyValueNet
    
    
    game_kwargs = {
        "M": 8,
        "N": 8,
        "K": 5,
    }
    
    game = Gomoku(**game_kwargs)
    game.set_play_only()
    
    net = PolicyValueNet(game_kwargs["M"], game_kwargs["N"])
    
    
    uct = UCT_Player(
        policy_kwargs={"C": 5},
        iterations=400,
        temp=.001,
    )
    
    uct2 = UCT_Player(
        policy_value_fn=net.policy_value_fn_sorted,
        policy_kwargs={"C": 5},
        iterations=400,
        temp=.001,
    )