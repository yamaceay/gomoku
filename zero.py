from src import ADP_Pre_Player, AlphaZeroPlayer, Gomoku

if __name__ == "__main__":
    game_kwargs = {
        'M': 8,
        'N': 8,
        'K': 5,
        'ADJ': 2,
    }
    
    game = Gomoku(**game_kwargs)
    player2 = AlphaZeroPlayer(**game_kwargs)
    player = ADP_Pre_Player(**game_kwargs)
    
    while not game.fin():
        move = player.next_move(game)
        game.play(move)
        print(game)
        # if game.fin():
        #     break
        # move = player2.next_move(game)
        # game.play(move)
        # print(game)