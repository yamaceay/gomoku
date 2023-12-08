from src import ADP_Pre_Player, Gomoku

import numpy as np

if __name__ == "__main__":
    from collections import deque
    game_kwargs = {
        'M': 8,
        'N': 8,
        'K': 5,
        'ADJ': 2,
    }
    
    buffer_size = 100
    buffer = deque(maxlen=buffer_size)
    
    for i in range(1):
        game = Gomoku(**game_kwargs)
        player = ADP_Pre_Player(**game_kwargs)
        
        move  = player.next_move(game)
        game.play(move)
        move = player.next_move(game)
        game.play(move)

        transformations = np.unravel_index(range(8), (4, 2, 2))
        for transformation in zip(*transformations):
            new_game = Gomoku(**game_kwargs)
            new_game.play(*game.history(*transformation))
            print(new_game)