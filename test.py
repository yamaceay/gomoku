import os
import argparse
import numpy as np
from src import Gomoku, UCT_Zero_Player, ADP_Dense_Player, ADP_Conv_Player, ADP_Pre_Player

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--PLAYER", type=str, help="Which model")
    parser.add_argument("--DIR", type=str, help="Directory where the model is located")
    parser.add_argument("--EPOCH", type=int, help="Epoch to be tested")
    parser.add_argument("--ITERATIONS", type=int, default=400, help="Number of UCT iterations")
    parser.add_argument("--EPSILON", type=float, default=.25, help="Policy exploration rate")
    parser.add_argument("--MAX_DEPTH", type=int, default=10, help="Maximum simulation depth")
    parser.add_argument("--SIM_IS_RANDOM", action="store_true", help="Whether to simulate randomly or using ADP")
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
    
    model_path = os.path.join(args.DIR, "models/epoch_{}.h5".format(args.EPOCH))

    adp_model = player(model_path=model_path, game_kwargs=game_kwargs)
    
    uct_adp = UCT_Zero_Player(
            adp_model=adp_model, 
            iterations=args.ITERATIONS, 
            max_depth=args.MAX_DEPTH,
            sim_is_random=args.SIM_IS_RANDOM,
            policy_kwargs={"C": .1},
    )
    
    game = Gomoku(**game_kwargs)
    while not game.fin():
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
        
    # results = eval_by_uct(game_kwargs, adp, adp, n_test_games=3, iterations=200, epsilon=.1)
    # print(results)
    
    # uct = UCT_Player(iterations=2000, policy=uct_score)
    # zero = AlphaZeroPlayer(game_kwargs)
    
    # depth = 8
    # for i in range(50):
    #     game = Gomoku(**game_kwargs)
    #     for i in tqdm(range(game.M * game.N)):                        
    #         if i < depth:
    #             # if i % 2 == 0:
    #             #     print("ZERO BEGIN")
    #             #     action = zero.next_move(game, epsilon=.25)
    #             #     print("ZERO END")
    #             # else:
    #             print("ADP BEGIN")
    #             action = adp.next_move(game, epsilon=.1)
    #             print("ADP END")
            
    #         else:
    #             print("UCT BEGIN")
    #             rewards_actions = uct.rewards_actions(game)
    #             print("UCT END")

    #             if game.player == -1:
    #                 _, action = rewards_actions[-1]
    #             else:
    #                 _, action = rewards_actions[0]
                    
    #             print("COLLECT BEGIN")
    #             for r, a in rewards_actions:
    #                 new_game = game.copy()
    #                 new_game.play(a)
    #                 buffer.extend([(s, r) for s, _ in collect_play_data(new_game)])
    #             print("COLLECT END", len(buffer))
                
    #             print("TRAIN BEGIN")
    #             sample = random.sample(buffer, min(400, len(buffer)))
    #             loss = adp.train_batch(sample, disable=False)
    #             print("TRAIN END", loss)
            
    #         game.play(action)
    #         print(game)
    #         if game.fin():
    #             break    
            
    #     adp.nn.save_model(f"_dens3/models/epoch_{i}.h5")