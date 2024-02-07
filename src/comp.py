from .gomoku import Gomoku, S_GAME, M_GAME, L_GAME
from .player import Player
from .data import play_game
from .net import Zero_Net
from .mcts import Deep_Player
from .hybrid import Flat_Player
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

def competition(
    game: Gomoku,
    n_games: int,
    player1_args: tuple[str, Player, float], 
    player2_args: tuple[str, Player, float],
    bar: tqdm = None,
    *args, **kwargs
) -> pd.DataFrame:
    
    name1, player1, epsilon1 = player1_args
    name2, player2, epsilon2 = player2_args
    
    fst_time_stats, snd_time_stats = [], []
    
    results = []
    for _ in range(n_games):
        new_game, curr_started, timeline = play_game(game, player1, player2, epsilon1, epsilon2, *args, **kwargs)
        
        timeline += [np.nan] * (game.M*game.N - len(timeline))
        fst = pd.Series(timeline[0::2])
        snd = pd.Series(timeline[1::2])
        
        score = new_game.score()
        if not curr_started:
            score *= -1
            fst, snd = snd, fst
            
        fst_time_stats += [fst]
        snd_time_stats += [snd]
        
        results.append({
            "score": score, 
            "curr_started": curr_started, 
            "length": len(new_game.history),
        })
        
        if bar is not None:
            bar.update(1)
    
    results = pd.DataFrame(results)  
    fst_time_stats, snd_time_stats = pd.DataFrame(fst_time_stats), pd.DataFrame(snd_time_stats)  
    
    indices = [f"{name1}_started", f"{name2}_started"] # True, False
    columns = [f"{name1}_won", "Tie", f"{name2}_won"] # 1, 0, -1

    results_counts = pd.DataFrame(index=indices, columns=columns)
    results_lengths = results.copy()

    results_counts.fillna(0, inplace=True)

    for column in columns:
        for index in indices:
            index_val = True if index == f"{name1}_started" else False
            column_val = 1 if column == f"{name1}_won" else -1 if column == f"{name2}_won" else 0
            
            filter_fn = lambda x: (x["curr_started"] == index_val) & (x["score"] == column_val)
            
            results_lengths[index + '_' + column] = results_lengths.apply(lambda x: x["length"] if filter_fn(x) else np.nan, axis=1)
            results_counts.loc[index, column] = len(results[filter_fn(results)]) / n_games
    
    results_lengths.drop(["curr_started", "score", "length"], axis=1, inplace=True)
    
    return results_counts, results_lengths, fst_time_stats, snd_time_stats
                
def append_csv(df: pd.DataFrame, path: str):
    if not os.path.exists(path):
        df.to_csv(path)
    else:
        df.to_csv(path, mode='a', header=False)

if __name__ == '__main__':
    (M, N, K) = CURR_GAME = S_GAME

    net = Zero_Net(
        game_kwargs=CURR_GAME, 
        model_file=f"bin/models/best_{M}_{N}_{K}.model",
    )

    players = [
        ("FLAT", Flat_Player(policy_value_fn=net.predict), .0),
        ("ZERO", Deep_Player(iterations=400, policy_value_fn=net.predict), .0),
        ("ZEROX", Deep_Player(iterations=400, policy_value_fn=net.predict, memory=True), .0),
        ("UCT1", Deep_Player(iterations=1000), .0),
        ("UCT3", Deep_Player(iterations=3000), .0),
        ("UCT6", Deep_Player(iterations=6000), .0),
    ]

    IMG_DIR = f"out/{M}_{N}_{K}"
    TIME_DIR = os.path.join(IMG_DIR, "timeseries")
    os.makedirs(TIME_DIR, exist_ok=True)
    COMP_DIR = os.path.join(IMG_DIR, "competition")
    os.makedirs(COMP_DIR, exist_ok=True)
    
    game = Gomoku(*CURR_GAME)
    n_games = 1
    st_ind = 0
    tested_players = players[st_ind:3]

    n_players = len(players)
    n_tested_players = len(tested_players)
    n_rival_players = n_players - n_tested_players

    n_rival_duels = n_tested_players * n_rival_players
    n_each_duels = 2 ** (n_tested_players - 1) - 1
    n_done_duels = 2 * st_ind
    total_duels = n_rival_duels + n_each_duels - n_done_duels

    for player in players:
        player_path = os.path.join(TIME_DIR, f"{player[0]}.csv")
        if not os.path.exists(player_path):
            time_stats = pd.DataFrame(columns=list(range(game.M*game.N)))
            time_stats.to_csv(player_path)

    with tqdm(
        range(total_duels * n_games), 
        desc="Competition",
        position=0,
        unit="game",
    ) as bar:
        for i in range(n_tested_players):
            tested_player = tested_players[i]
            for j in range(i + st_ind + 1, n_players):
                rival_player = players[j]
                
                comp_results = competition(
                    game, n_games, 
                    tested_player, rival_player, 
                    fairness=.5, 
                    timeline=True, 
                    bar=bar,
                )
                
                results_counts, results_lengths, fst_time_stats, snd_time_stats = comp_results
                
                append_csv(fst_time_stats, os.path.join(TIME_DIR, f"{tested_player[0]}.csv"))
                append_csv(snd_time_stats, os.path.join(TIME_DIR, f"{rival_player[0]}.csv"))
                
                append_csv(results_counts, os.path.join(COMP_DIR, f"{tested_player[0]}_{rival_player[0]}_counts.csv"))
                append_csv(results_lengths, os.path.join(COMP_DIR, f"{tested_player[0]}_{rival_player[0]}_lengths.csv"))
