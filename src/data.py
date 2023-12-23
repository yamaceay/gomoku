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

def play_until_end_zero(
    game: Gomoku, 
    player1: UCT_Player = None, 
    player2: UCT_Player = None,
    epsilon1: float = .0,
    epsilon2: float = .0,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], bool]:
    
    if game.fin() or (player1 is None and player2 is None):
        return [], True
        
    list_states = []
    list_probs_actions = []
    new_game = game.copy()
    
    if player2 is None:
        while not new_game.fin():
            action, probs_actions = player1.next_move(
                new_game, 
                epsilon=epsilon1, 
                probs=True,
            )
            list_states += [new_game.to_zero_input()]
            list_probs_actions += [probs_actions]
            new_game.play(action)
        winner = new_game.score()
        list_winner = np.ones(len(list_probs_actions)) * winner
        play_data = zip(list_states, list_probs_actions, list_winner)
        return play_data, True    
    
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


def collect_play_data_zero(
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
        # for feature in new_game.history_str_aug():
        #     label = new_game.score()
        #     play_data += [(feature, label)]
    
    return play_data