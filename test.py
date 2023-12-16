import os
import argparse
import numpy as np
from src import Gomoku, AlphaZeroPlayer, UCT_Tang_Player, UCT_Player, ADP_Dense_Player, ADP_Conv_Player, ADP_Pre_Player

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--PLAYER", type=str, help="Which model")
    parser.add_argument("--DIR", type=str, help="Directory where the model is located")
    parser.add_argument("--EPOCH", type=int, help="Epoch to be tested")
    parser.add_argument("--ITERATIONS", type=int, default=400, help="Number of UCT iterations")
    parser.add_argument("--EPSILON", type=float, default=.25, help="Policy exploration rate")
    # parser.add_argument("--MAX_DEPTH", type=int, default=10, help="Maximum simulation depth")
    # parser.add_argument("--SIM_IS_RANDOM", action="store_true", help="Whether to simulate randomly or using ADP")
    args = parser.parse_args()
    
    players = {
        "dense": ADP_Dense_Player,
        "conv": ADP_Conv_Player,
        "pre": ADP_Pre_Player,
    }

    player = players[args.PLAYER]

    game_kwargs = {
        "M": 8,
        "N": 8,
        "K": 5,
        "ADJ": 2,
    }
    
    uct_model = UCT_Player(
        iterations = args.ITERATIONS / 5,
    )
    
    model_path = os.path.join(args.DIR, "models/epoch_{}.h5".format(args.EPOCH))

    adp_model = player(model_path=model_path, game_kwargs=game_kwargs)
    zero = AlphaZeroPlayer(game_kwargs)
    
    uct_adp = UCT_Tang_Player(
            adp_model=adp_model,
            uct_model=uct_model,
            k=5,
            C_ADP=1,
    )
    
    game = Gomoku(**game_kwargs)
    while not game.fin():
        action = zero.next_move(game)
        game.play(action)
        print(game)
        if game.fin():
            break

        probs_actions = uct_adp.next_move_probs(game)
        print("\n".join([f"{a} {p:.4f}" for p, a in probs_actions]))
        
        probs, actions = zip(*probs_actions)
        if np.random.random() < args.EPSILON:
            action_i = np.random.choice(range(len(actions)), p=probs)
            action = actions[action_i]
        else:
            action = actions[0]
        game.play(action)
        print(game)