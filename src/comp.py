from .gomoku import Gomoku, S_GAME, M_GAME, L_GAME
from .player import Player
from .data import play_game
from .net import Zero_Net
from .mcts import Deep_Player
from .hybrid import Flat_Player
from .train import TRAIN_ARGS
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="Compare players")
parser.add_argument("--game", "-g", type=str, choices=["S", "M", "L"], help="game size")
args = parser.parse_args()
game_size = args.game

game_kwargs = (M, N, K) = S_GAME if game_size == "S" else M_GAME if game_size == "M" else L_GAME
game_kwargs_str = f"{M}_{N}_{K}"

assert game_kwargs_str in TRAIN_ARGS, f"stringified game kwargs must be in {list(TRAIN_ARGS.keys())}"
train_params = TRAIN_ARGS[game_kwargs_str]
    
n_zero = train_params["n_zero"]
n_uct = train_params["n_uct"]
n_uct_step = train_params["n_uct_step"]
n_uct_max = train_params["n_uct_max"]

IMG_DIR = f"out/{game_kwargs_str}"
TIME_DIR = os.path.join(IMG_DIR, "timeseries")
os.makedirs(TIME_DIR, exist_ok=True)
COMP_DIR = os.path.join(IMG_DIR, "competition")
os.makedirs(COMP_DIR, exist_ok=True)

class Comparator:
    def __init__(self,
                 game: Gomoku = Gomoku(*game_kwargs),
                 n_games: int = 50,
                 ):
        
        self.game = game
        self.n_games = n_games
        
    def comp(self, 
             players: list[tuple[str, Player, float]], 
             edges: list[tuple[int, int]] = None
             ):
        
        for player in players:
            player_path = os.path.join(TIME_DIR, f"{player[0]}.csv")
            if not os.path.exists(player_path):
                time_stats = pd.DataFrame(columns=list(range(M*N)))
                time_stats.to_csv(player_path)

        with tqdm(
            range(len(edges) * self.n_games), 
            desc="Competition",
            position=0,
            unit="game",
        ) as bar:
            for i, j in edges:
                tested_player, rival_player = players[i], players[j]
                    
                comp_results = competition(
                    self.game, self.n_games, 
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
    net = Zero_Net(
        game_kwargs=game_kwargs, 
        model_file=f"bin/models/best_{game_kwargs_str}.model",
    )

    players = [
        ("FLAT", Flat_Player(policy_value_fn=net.predict), .0),
        ("ZERO", Deep_Player(iterations=n_zero, policy_value_fn=net.predict), .0),
    ]

    for n_uct_it in range(n_uct_step, n_uct_max + n_uct_step, n_uct_step):
        players += [
            (f"UCT_{n_uct_it}", Deep_Player(iterations=n_uct_it), .0)    
        ]
        
    st_ind, nd_ind = 0, 2
    
    edges = []
    for i in range(st_ind, nd_ind):
        for j in range(i + 1, len(players)):
            edges += [(i, j)]
            
    comparator = Comparator()
    comparator.comp(players, edges)
