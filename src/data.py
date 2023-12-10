from .gomoku import Gomoku
from .players import Player
from tqdm import tqdm
import random
    
def collect_selfplay_data(
    player: Player, 
    game_kwargs: dict[int], 
    n_games: int = 1, 
    epsilon: float = .1) -> list[str]:
    
    selfplay_data = []
    for _ in tqdm(range(n_games), 
                  position=1, 
                  leave=False, 
                  desc="Collecting self-play data"
                  ):
        game = Gomoku(**game_kwargs)
        while not game.fin():
            action = game.actions()[0]
            if random.random() >= epsilon:
                action = player.next_move(game)
            game.play(action)
        for transformation in game.transformations:
            selfplay_data += [game.history_str(*transformation)]
    return selfplay_data

def play_until_end(
    game_kwargs: dict[int], 
    player1: Player, 
    player2: Player,
    epsilon1: float = .0,
    epsilon2: float = .0,
    ) -> tuple[Gomoku, bool]:
    
    game = Gomoku(**game_kwargs)
    player2_starts = random.random() < .5
    if player2_starts:
        action = game.actions()[0]
        if random.random() >= epsilon2:
            action = player2.next_move(game)
        game.play(action)
    
    while not game.fin():
        action = game.actions()[0]
        if random.random() >= epsilon1:
            action = player1.next_move(game)
        game.play(action)
        if game.fin():
            break
        action = game.actions()[0]
        if random.random() >= epsilon2:
            action = player2.next_move(game)
        game.play(action)

    return game, not player2_starts

def collect_play_data(
    game_kwargs: dict[int], 
    learner: Player,
    trainer: Player, 
    n_games: int = 1, 
    epsilon: float = .1) -> list[str]:
    
    learner_args = {
        "player1": learner,
        "epsilon1": epsilon,
    }
    
    trainer_args = {
        "player1": trainer,
        "epsilon1": .0,
    }
        
    play_data = []        
    for _ in tqdm(range(n_games), 
                  position=1, 
                  leave=False,
                  desc="Collecting play data"
                  ):
        game, _ = play_until_end(game_kwargs, **learner_args, **trainer_args)
        play_data += [game.history_str()]
    
    return play_data