from .gomoku import Gomoku
from .players import Player
from tqdm import tqdm
import random

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
    
    play_data = []
    for _ in tqdm(range(n_games), 
                  position=1, 
                  leave=False, 
                  desc="Collecting play data"
                  ):
        
        game, _ = play_until_end(game, **learner_args, **trainer_args)
        for transformation in game.transformations:
            feature = game.history_str(*transformation)
            label = game.score()
            play_data += [(feature, label)]
    return play_data