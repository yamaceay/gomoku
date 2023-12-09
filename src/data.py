from .gomoku import Gomoku
from .players import Player
from tqdm import tqdm
import numpy as np
    
def collect_selfplay_data(player: Player, game_kwargs: dict[int], n_games: int = 1) -> list[str]:
    selfplay_data = []
    for _ in tqdm(range(n_games)):
        game = Gomoku(**game_kwargs)
        while not game.fin():
            action = player.next_move(game)
            game.play(action)
        for transformation in game.transformations:
            selfplay_data += [game.history_str(*transformation)]
    return selfplay_data

if __name__ == "__main__":
    from collections import deque
    from .adp import ADP_Pre_Player
    
    game_kwargs = {
        'M': 8,
        'N': 8,
        'K': 5,
        'ADJ': 2,
    }
    
    buffer_size = 100
    buffer = deque(maxlen=buffer_size)
    
    player = ADP_Pre_Player(**game_kwargs)
    selfplay_data = collect_selfplay_data(player, game_kwargs, n_games=3)
    
    buffer.extend(selfplay_data)
    print(buffer)
