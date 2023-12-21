import logging
import os
from src import train_adp, ADP_Dense_Player, ADP_Pre_Player, ADP_Conv_Player

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--DIR', type=str, help='Directory path')
    parser.add_argument('--PLAYER', type=str, help='Player type')

    # epochs 
    parser.add_argument('--START', type=int, help='Start epoch')
    parser.add_argument('--END', type=int, help='End epoch')
    parser.add_argument('--STEP', type=int, help='Epoch step')
    parser.add_argument('--CHECKPOINT', type=int, default=50, help='Checkpoint epoch')
    parser.add_argument('--NO_EVAL', action='store_true', help='Evaluate or not')
    parser.add_argument('--NO_TRAIN', action='store_true', help='Train or not')
    
    # training
    parser.add_argument('--BATCH_SIZE', type=int, help='Batch size')
    parser.add_argument('--ZERO_PLAY', action='store_true', help='Learn by playing against Zero')
    parser.add_argument('--SELECT_BEST', action='store_true', help='Select best model or not')
    parser.add_argument('--LR_DECAY', type=float, default=0.99, help='Learning rate decay')
    parser.add_argument('--N_TEST_GAMES', type=int, default=7, help='Number of games to play against Zero')
    parser.add_argument('--BUFFER_SIZE', type=int, default=10000, help='Buffer size')
    
    # eval
    parser.add_argument('--EVAL_ITERATIONS', type=int, default=200, help='Number of iterations to evaluate')
    parser.add_argument("--EVAL_MAX_DEPTH", type=int, default=10, help="Maximum simulation depth during evaluation")
    parser.add_argument("--EVAL_SIM_IS_RANDOM", action="store_true", help="Whether to simulate randomly or using ADP during evaluation")
    
    # game
    parser.add_argument('--M', type=int, default=8, help='Board width')
    parser.add_argument('--N', type=int, default=8, help='Board height')
    parser.add_argument('--K', type=int, default=5, help='Number of stones to align')
    parser.add_argument('--ADJ', type=int, default=0, help='Number of adjacent stones to consider')
    
    # adp args
    parser.add_argument('--ALPHA', type=float, default=0.9, help='Q-Learning rate')
    parser.add_argument('--GAMMA', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--LR', type=float, default=0.01, help='NN Learning rate')
    parser.add_argument('--EPSILON', type=float, default=0.1, help='Policy exploration rate')
        
    args = parser.parse_args()

    LOSSES_PATH = os.path.join(args.DIR, "logs/losses.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    file_handler = logging.FileHandler(LOSSES_PATH)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    game_kwargs = {
        'M': args.M,
        'N': args.N,
        'K': args.K,
    }
    if args.ADJ > 0:
        game_kwargs['ADJ'] = args.ADJ
    
    player_kwargs = {
        'alpha': args.ALPHA,
        'gamma': args.GAMMA,
        'lr': args.LR,
        'logger': logger,
    }
    
    lr_kwargs = {
        'lr_decay': args.LR_DECAY,
    }
   
    player_types = {
        'dense': ADP_Dense_Player,
        'pre': ADP_Pre_Player,
        'conv': ADP_Conv_Player,
    }
    
    assert args.PLAYER in player_types, f"Player type {args.PLAYER} not found"
    player = player_types.get(args.PLAYER)
    
    train_adp(
        dir_path=args.DIR,
        player=player,
        
        epochs_start=args.START,
        epochs_end=args.END,
        epochs_step=args.STEP,
        eval=not args.NO_EVAL,
        train=not args.NO_TRAIN,
        checkpoint=args.CHECKPOINT,
        
        batch_size=args.BATCH_SIZE,
        zero_play=args.ZERO_PLAY,
        select_best=args.SELECT_BEST,
        lr_args=lr_kwargs,
        eval_n_games=args.N_TEST_GAMES,
        buffer_size=args.BUFFER_SIZE,
        
        eval_iterations=args.EVAL_ITERATIONS,
        eval_max_depth=args.EVAL_MAX_DEPTH,
        eval_sim_is_random=args.EVAL_SIM_IS_RANDOM,
        
        game_kwargs=game_kwargs,
         
        player_args=player_kwargs,
        epsilon=args.EPSILON,
    )
            