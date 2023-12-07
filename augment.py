from src import ADP_Pre_Player, AlphaZeroPlayer, Gomoku

import numpy as np

def augment_data(game: Gomoku):
    if game.last_move is None:
        return [game]
    
    board, last_move = game.board, game.last_move
    mboard = np.zeros_like(board)
    mboard[last_move] = 1
    
    rots, lrs, uds = np.unravel_index(range(8), (2, 2, 2))

    augmented_data = []
    for keys in zip(rots, lrs, uds):
        rot, lr, ud = keys
        new_board, new_mboard = np.array(board), np.array(mboard)
        if rot:
            new_mboard = np.rot90(new_mboard, k=rot)
            new_board = np.rot90(new_board, k=rot)
        if lr:
            new_mboard = np.fliplr(new_mboard)
            new_board = np.fliplr(new_board)
        if ud:
            new_mboard = np.flipud(new_mboard)
            new_board = np.flipud(new_board)

        last_move_x, last_move_y = np.where(new_mboard == 1)
        last_move = (int(last_move_x[0]), int(last_move_y[0]))
        
        new_game = game.copy()
        new_game.set_last_move(last_move)
        new_game.board = new_board
        
        augmented_data.append((keys, new_game))
    
    return augmented_data

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
        
        while not game.fin():
            move  = player.next_move(game)
            game.play(move)
            move = player.next_move(game)
            game.play(move)
            for keys, state in augment_data(game):
                print(keys, state, state.last_move)
            break