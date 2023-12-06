import json
import os

from IPython.display import clear_output
from tqdm import tqdm

from src import comp_models, Gomoku, UCT_Player, Player, uct_score, uct_pb_score

def my_print(*args, **kwargs):
    clear_output(wait=True)
    print(*args, **kwargs)

def play_game(game: Gomoku, player1: Player, player2: Player, verbose: int = 0):
    while not game.fin():
        move = player1.next_move(game)
        game.play(move)
        if verbose > 0:
            if verbose > 1:
                print(player1.next_move_probs(game))
            print(game)
        if game.fin():
            break
        move = player2.next_move(game)
        game.play(move)
        if verbose > 0:
            if verbose > 1:
                print(player2.next_move_probs(game))
            print(game)
    return game.winner

def play_n_games(game: Gomoku, player1: Player, player2: Player, n: int, verbose: int = 0):
    avg = 0
    avgs = []
    for i in range(n):
        score = (play_game(game.copy(), player1, player2, verbose) + 1) / 2
        avg *= i / (i + 1)
        avg += score / (i + 1)
        avgs += [avg]
    return avgs

def tournament(game_kwargs, models: list[Player], n_test_games: int, start_ind: int = 0) -> list[tuple[int, int, int, int]]:
    total_ind = 0
    with tqdm(
        total=n_test_games * len(models) * (len(models) - 1) // 2, 
        position=0, 
        leave=False, 
        desc="Tournament"
    ) as bar:
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                if total_ind < start_ind:
                    bar.update(n_test_games)
                    total_ind += n_test_games
                    continue
                n_wins, l_history = 0, 0
                for _ in range(n_test_games):
                    win, starts, len_history = comp_models(game_kwargs, models[i], models[j])
                    n_wins += (win > 0) == starts
                    l_history += len_history
                    bar.update(1)
                    total_ind += 1
                n_wins, l_history = n_wins / n_test_games, l_history / n_test_games
                yield (i, j, n_wins, l_history)
    
if __name__ == "__main__": 
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    file_handler = logging.FileHandler("_mcts/logs/results.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    
    game_kwargs = {
        'M': 5,
        'N': 5,
        'K': 4,
        'ADJ': 2,
    }

    n_its = [100, 300, 900]
    c_pbs = [0.1, 0.5, 2.5]

    players = [
        UCT_Player(
            iterations=n_it, 
            policy=uct_pb_score, 
            policy_kwargs={
                "C": 1,
                "C_PB": c_pb,
            }, 
        )
        for c_pb in c_pbs
        for n_it in n_its
    ]

    for (i, j, n_wins, l_history) in tournament(game_kwargs, players, 5, 0):
        n_it_i = n_its[i // len(c_pbs)]
        c_pb_i = c_pbs[i % len(c_pbs)]
        n_it_j = n_its[j // len(c_pbs)]
        c_pb_j = c_pbs[j % len(c_pbs)]
        
        results = {
            "n_it_i": n_it_i,
            "c_pb_i": c_pb_i,
            "n_it_j": n_it_j,
            "c_pb_j": c_pb_j,
            "n_wins": n_wins,
            "l_history": l_history,
        }
        
        logger.info(json.dumps(results))