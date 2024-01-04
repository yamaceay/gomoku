from .gomoku import Gomoku
from .data import play_game
from .player import Player
from tqdm import tqdm

def competition(
    game: Gomoku,
    n_games: int,
    player1: Player, 
    player2: Player, 
    epsilon1: float = .0,
    epsilon2: float = .0,
    fairness: float = .5,
    verbose: bool = False,
) -> pd.DataFrame:
    
    pbar = range(n_games)
    if verbose:
        pbar = tqdm(pbar, desc="Competition")
    
    results = []
    for _ in pbar:
        new_game, curr_started = play_game(game, player1, player2, epsilon1, epsilon2, fairness, verbose)
        score = new_game.score()
        if not curr_started:
            score *= -1
        results.append({"score": score, "curr_started": curr_started})
    return pd.DataFrame(results)