from .gomoku import Gomoku
from .player import Player
from tqdm import tqdm
import random
from .mcts import Deep_Player
import numpy as np
from time import time


def play_game(
    game: Gomoku, 
    player1: Player = None, 
    player2: Player = None,
    epsilon1: float = .0,
    epsilon2: float = .0,
    fairness: float = .5,
    verbose: bool = False,
    timeline: bool = False,
    ) -> tuple[Gomoku, bool, list[float]]:
    
    if game.fin() or (player1 is None and player2 is None):
        return game, True, []
        
    time_list = []
    new_game = game.copy()
    
    if player2 is None:
        while not new_game.fin():
            start = time()
            action = player1.next_move(new_game, epsilon=epsilon1)
            if timeline:
                time_list += [time() - start]
            new_game.play(action)
            if verbose:
                print(new_game)
        return new_game, True    
    
    player2_starts = random.random() < fairness
    if player2_starts:
        start = time()
        action = player2.next_move(new_game, epsilon=epsilon2)
        if timeline:
            time_list += [time() - start]
        new_game.play(action)
        if verbose:
            print(new_game)
    
    while not new_game.fin():
        start = time()
        action = player1.next_move(new_game, epsilon=epsilon1)
        if timeline:
            time_list += [time() - start]
        new_game.play(action)
        if verbose:
            print(new_game)
        if new_game.fin():
            break
        start = time()
        action = player2.next_move(new_game, epsilon=epsilon2)
        if timeline:
            time_list += [time() - start]
        new_game.play(action)
        if verbose:
            print(new_game)

    return new_game, not player2_starts, time_list

def play_game_for_train(
    game: Gomoku, 
    player: Deep_Player = None, 
    epsilon: float = .0,
    verbose: bool = False,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    
    if game.fin() or player is None:
        return []
        
    list_states = []
    list_probs = []
    new_game = game.copy()
    
    game_bar = range(game.M * game.N)
    if verbose:
        game_bar = tqdm(game_bar)
        
    for _ in game_bar:
        if new_game.fin():
            break
        list_states += [new_game.encode()]
        
        action, probs = player.next_move_for_train(
            new_game, 
            epsilon=epsilon,
        )
        
        list_probs += [probs]
        new_game.play(action)

    winner = new_game.score()
    list_winner = np.ones(len(list_probs)) * winner
    return zip(list_states, list_probs, list_winner)  


def play_n_games_for_train(
    game: Gomoku,
    n_games: int = 1, 
    player: Deep_Player = None,
    epsilon: float = .0,
    ) -> list[tuple[str, float]]:
    
    return [
        (state, probs, winner)
        for state, probs, winner in play_game_for_train(
            game, 
            player=player, 
            epsilon=epsilon
        ) 
        for _ in tqdm(
            range(n_games), 
            position=1, 
            leave=False, 
            desc="Collecting self-play data",
            disable=True
        )
    ]

def extend_play_data(
    play_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    
    extend_data = []
    for state, mcts_prob, winner in play_data:
        _, m, n = state.shape
        for i in range(1, 5):
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