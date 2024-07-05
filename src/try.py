from .gomoku import Gomoku, S_GAME, M_GAME, L_GAME, Pattern
from .comp import get_player

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game_size", type=str, choices=["S", "M", "L"], required=True, help="Game size specified as (M,N,K). S=(6,6,4), M=(8,8,5), L=(10,10,5)") # game size
    parser.add_argument("-p", "--player", choices=["UCT", "FLAT", "ZERO", "ZEROX"], required=True, help="Player type. UCT: Vanilla MCTS, FLAT: AlphaZero w/o MCTS, ZERO: AlphaZero, ZEROX: AlphaZero w/ memory.")
    parser.add_argument("--epsilon", type=float, default=.0, help="Noise parameter for any player, ranging from 0 to 1. Zero for deterministic players, one for random players. Defaults to 0.")
    parser.add_argument("--level", type=int, default=0, help="Strength of player, represented by 1, ..., MAX_LEVEL (= 0). Defaults to 0. If player type is UCT, you must specify level.")
    parser.add_argument("-s", "--silent", action="store_true") # true for API calls, don't modify
    args = parser.parse_args()
    game_kwargs = S_GAME if args.game_size == "S" else M_GAME if args.game_size == "M" else L_GAME
    (name, player, _), _ = get_player(game_kwargs, args.player)
    if not args.silent:
        print(f"Player: {name}")
    
    game = Gomoku(*game_kwargs)
    try:
        while not game.fin():
            if not args.silent:
                print(game)
                action_str = str(input("Enter move: "))
            else:
                action_str = str(input())
            (i,j) = Pattern.loc_to_move_one(action_str)
            action = (i, j+1)
            game.play(action)
            if game.fin():
                break
            action = player.next_move(game, args.epsilon)
            if args.silent:
                (i, j) = action
                action_str = Pattern.move_to_loc((i, j-1))
                print(action_str)
            game.play(action)
    except KeyboardInterrupt:
        print('\n\rStopped')
    if not args.silent:
        print(game)