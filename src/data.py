from .gomoku import Gomoku
from .players import Player
from .adp import ADP_Player
from .patterns import Pattern
from tqdm import tqdm
from collections import deque
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def collect_selfplay_data(
    player: Player, 
    game_kwargs: dict[int], 
    n_games: int = 1, 
    epsilon: float = .1) -> list[str]:
    
    selfplay_data = []
    for _ in tqdm(range(n_games)):
        game = Gomoku(**game_kwargs)
        while not game.fin():
            action = game.actions()[0]
            if random.random() >= epsilon:
                action = player.next_move(game)
            game.play(action)
        for transformation in game.transformations:
            selfplay_data += [game.history_str(*transformation)]
    return selfplay_data

def play_until_end(
    game_kwargs: dict[int], 
    player1: Player, 
    player2: Player,
    epsilon1: float = .0,
    epsilon2: float = .0,
    ) -> tuple[Gomoku, bool]:
    
    game = Gomoku(**game_kwargs)
    player2_starts = random.random() < .5
    if player2_starts:
        action = game.actions()[0]
        if random.random() >= epsilon2:
            action = player2.next_move(game)
        game.play(action)
    
    while not game.fin():
        action = game.actions()[0]
        if random.random() >= epsilon1:
            action = player1.next_move(game)
        game.play(action)
        if game.fin():
            break
        action = game.actions()[0]
        if random.random() >= epsilon2:
            action = player2.next_move(game)
        game.play(action)

    return game, not player2_starts

def collect_play_data(
    game_kwargs: dict[int], 
    learner: Player,
    trainer: Player, 
    n_games: int = 1, 
    epsilon: float = .1) -> list[str]:
    
    learner_args = {
        "player1": learner,
        "epsilon1": epsilon,
    }
    
    trainer_args = {
        "player1": trainer,
        "epsilon1": .0,
    }
        
    play_data = []        
    for _ in tqdm(range(n_games)):
        game, _ = play_until_end(game_kwargs, **learner_args, **trainer_args)
        play_data += [game.history_str()]
    
    return play_data

if __name__ == "__main__":
    from .adp import ADP_Pre_Player
    
    game_kwargs = {
        'M': 8,
        'N': 8,
        'K': 5,
        'ADJ': 2,
    }
    
    buffer_size = 1000
    buffer = deque(maxlen=buffer_size)
    
    buffer.extend(['c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4', 
                   'a6,a5,b5,c4,c5,b4,a3,a2,e4,a8,f3,g6,a4,d2,c2,e2,g4,f4,d4,d6,d7,e8,h7,g7,g8,e5,e6,h6,e7,c3,c6,e1', 
                   'h6,h5,g5,f4,f5,g4,h3,h2,d4,h8,c3,b6,h4,e2,f2,d2,b4,c4,e4,e6,e7,d8,a7,b7,b8,d5,d6,a6,d7,f3,f6,d1', 
                   'a3,a4,b4,c5,c4,b5,a6,a7,e5,a1,f6,g3,a5,d7,c7,e7,g5,f5,d5,d3,d2,e1,h2,g2,g1,e4,e3,h3,e2,c6,c3,e8', 
                   'h3,h4,g4,f5,f4,g5,h6,h7,d5,h1,c6,b3,h5,e7,f7,d7,b5,c5,e5,e3,e2,d1,a2,b2,b1,d4,d3,a3,d2,f6,f3,d8', 
                   'c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4', 
                   'a6,a5,b5,c4,c5,b4,a3,a2,e4,a8,f3,g6,a4,d2,c2,e2,g4,f4,d4,d6,d7,e8,h7,g7,g8,e5,e6,h6,e7,c3,c6,e1', 
                   'h6,h5,g5,f4,f5,g4,h3,h2,d4,h8,c3,b6,h4,e2,f2,d2,b4,c4,e4,e6,e7,d8,a7,b7,b8,d5,d6,a6,d7,f3,f6,d1', 
                   'a3,a4,b4,c5,c4,b5,a6,a7,e5,a1,f6,g3,a5,d7,c7,e7,g5,f5,d5,d3,d2,e1,h2,g2,g1,e4,e3,h3,e2,c6,c3,e8', 
                   'h3,h4,g4,f5,f4,g5,h6,h7,d5,h1,c6,b3,h5,e7,f7,d7,b5,c5,e5,e3,e2,d1,a2,b2,b1,d4,d3,a3,d2,f6,f3,d8', 
                   'c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4', 
                   'a6,a5,b5,c4,c5,b4,a3,a2,e4,a8,f3,g6,a4,d2,c2,e2,g4,f4,d4,d6,d7,e8,h7,g7,g8,e5,e6,h6,e7,c3,c6,e1', 
                   'h6,h5,g5,f4,f5,g4,h3,h2,d4,h8,c3,b6,h4,e2,f2,d2,b4,c4,e4,e6,e7,d8,a7,b7,b8,d5,d6,a6,d7,f3,f6,d1', 
                   'a3,a4,b4,c5,c4,b5,a6,a7,e5,a1,f6,g3,a5,d7,c7,e7,g5,f5,d5,d3,d2,e1,h2,g2,g1,e4,e3,h3,e2,c6,c3,e8', 
                   'h3,h4,g4,f5,f4,g5,h6,h7,d5,h1,c6,b3,h5,e7,f7,d7,b5,c5,e5,e3,e2,d1,a2,b2,b1,d4,d3,a3,d2,f6,f3,d8', 
                   'c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4', 
                   'a6,a5,b5,c4,c5,b4,a3,a2,e4,a8,f3,g6,a4,d2,c2,e2,g4,f4,d4,d6,d7,e8,h7,g7,g8,e5,e6,h6,e7,c3,c6,e1', 
                   'h6,h5,g5,f4,f5,g4,h3,h2,d4,h8,c3,b6,h4,e2,f2,d2,b4,c4,e4,e6,e7,d8,a7,b7,b8,d5,d6,a6,d7,f3,f6,d1', 
                   'a3,a4,b4,c5,c4,b5,a6,a7,e5,a1,f6,g3,a5,d7,c7,e7,g5,f5,d5,d3,d2,e1,h2,g2,g1,e4,e3,h3,e2,c6,c3,e8', 
                   'h3,h4,g4,f5,f4,g5,h6,h7,d5,h1,c6,b3,h5,e7,f7,d7,b5,c5,e5,e3,e2,d1,a2,b2,b1,d4,d3,a3,d2,f6,f3,d8', 
                   'c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4', 
                   'a6,a5,b5,c4,c5,b4,a3,a2,e4,a8,f3,g6,a4,d2,c2,e2,g4,f4,d4,d6,d7,e8,h7,g7,g8,e5,e6,h6,e7,c3,c6,e1', 
                   'h6,h5,g5,f4,f5,g4,h3,h2,d4,h8,c3,b6,h4,e2,f2,d2,b4,c4,e4,e6,e7,d8,a7,b7,b8,d5,d6,a6,d7,f3,f6,d1', 
                   'a3,a4,b4,c5,c4,b5,a6,a7,e5,a1,f6,g3,a5,d7,c7,e7,g5,f5,d5,d3,d2,e1,h2,g2,g1,e4,e3,h3,e2,c6,c3,e8', 
                   'h3,h4,g4,f5,f4,g5,h6,h7,d5,h1,c6,b3,h5,e7,f7,d7,b5,c5,e5,e3,e2,d1,a2,b2,b1,d4,d3,a3,d2,f6,f3,d8', 
                   'c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4', 
                   'a6,a5,b5,c4,c5,b4,a3,a2,e4,a8,f3,g6,a4,d2,c2,e2,g4,f4,d4,d6,d7,e8,h7,g7,g8,e5,e6,h6,e7,c3,c6,e1', 
                   'h6,h5,g5,f4,f5,g4,h3,h2,d4,h8,c3,b6,h4,e2,f2,d2,b4,c4,e4,e6,e7,d8,a7,b7,b8,d5,d6,a6,d7,f3,f6,d1', 
                   'a3,a4,b4,c5,c4,b5,a6,a7,e5,a1,f6,g3,a5,d7,c7,e7,g5,f5,d5,d3,d2,e1,h2,g2,g1,e4,e3,h3,e2,c6,c3,e8', 
                   'h3,h4,g4,f5,f4,g5,h6,h7,d5,h1,c6,b3,h5,e7,f7,d7,b5,c5,e5,e3,e2,d1,a2,b2,b1,d4,d3,a3,d2,f6,f3,d8', 
                   'c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4', 
                   'a6,a5,b5,c4,c5,b4,a3,a2,e4,a8,f3,g6,a4,d2,c2,e2,g4,f4,d4,d6,d7,e8,h7,g7,g8,e5,e6,h6,e7,c3,c6,e1', 
                   'h6,h5,g5,f4,f5,g4,h3,h2,d4,h8,c3,b6,h4,e2,f2,d2,b4,c4,e4,e6,e7,d8,a7,b7,b8,d5,d6,a6,d7,f3,f6,d1', 
                   'a3,a4,b4,c5,c4,b5,a6,a7,e5,a1,f6,g3,a5,d7,c7,e7,g5,f5,d5,d3,d2,e1,h2,g2,g1,e4,e3,h3,e2,c6,c3,e8', 
                   'h3,h4,g4,f5,f4,g5,h6,h7,d5,h1,c6,b3,h5,e7,f7,d7,b5,c5,e5,e3,e2,d1,a2,b2,b1,d4,d3,a3,d2,f6,f3,d8', 
                   'c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4', 
                   'a6,a5,b5,c4,c5,b4,a3,a2,e4,a8,f3,g6,a4,d2,c2,e2,g4,f4,d4,d6,d7,e8,h7,g7,g8,e5,e6,h6,e7,c3,c6,e1', 
                   'h6,h5,g5,f4,f5,g4,h3,h2,d4,h8,c3,b6,h4,e2,f2,d2,b4,c4,e4,e6,e7,d8,a7,b7,b8,d5,d6,a6,d7,f3,f6,d1', 
                   'a3,a4,b4,c5,c4,b5,a6,a7,e5,a1,f6,g3,a5,d7,c7,e7,g5,f5,d5,d3,d2,e1,h2,g2,g1,e4,e3,h3,e2,c6,c3,e8', 
                   'h3,h4,g4,f5,f4,g5,h6,h7,d5,h1,c6,b3,h5,e7,f7,d7,b5,c5,e5,e3,e2,d1,a2,b2,b1,d4,d3,a3,d2,f6,f3,d8', 
                   'c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4', 
                   'a6,a5,b5,c4,c5,b4,a3,a2,e4,a8,f3,g6,a4,d2,c2,e2,g4,f4,d4,d6,d7,e8,h7,g7,g8,e5,e6,h6,e7,c3,c6,e1', 
                   'h6,h5,g5,f4,f5,g4,h3,h2,d4,h8,c3,b6,h4,e2,f2,d2,b4,c4,e4,e6,e7,d8,a7,b7,b8,d5,d6,a6,d7,f3,f6,d1', 
                   'a3,a4,b4,c5,c4,b5,a6,a7,e5,a1,f6,g3,a5,d7,c7,e7,g5,f5,d5,d3,d2,e1,h2,g2,g1,e4,e3,h3,e2,c6,c3,e8', 
                   'h3,h4,g4,f5,f4,g5,h6,h7,d5,h1,c6,b3,h5,e7,f7,d7,b5,c5,e5,e3,e2,d1,a2,b2,b1,d4,d3,a3,d2,f6,f3,d8', 
                   'c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4', 
                   'a6,a5,b5,c4,c5,b4,a3,a2,e4,a8,f3,g6,a4,d2,c2,e2,g4,f4,d4,d6,d7,e8,h7,g7,g8,e5,e6,h6,e7,c3,c6,e1', 
                   'h6,h5,g5,f4,f5,g4,h3,h2,d4,h8,c3,b6,h4,e2,f2,d2,b4,c4,e4,e6,e7,d8,a7,b7,b8,d5,d6,a6,d7,f3,f6,d1', 
                   'a3,a4,b4,c5,c4,b5,a6,a7,e5,a1,f6,g3,a5,d7,c7,e7,g5,f5,d5,d3,d2,e1,h2,g2,g1,e4,e3,h3,e2,c6,c3,e8', 
                   'h3,h4,g4,f5,f4,g5,h6,h7,d5,h1,c6,b3,h5,e7,f7,d7,b5,c5,e5,e3,e2,d1,a2,b2,b1,d4,d3,a3,d2,f6,f3,d8', 
                   'c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4', 
                   'a6,a5,b5,c4,c5,b4,a3,a2,e4,a8,f3,g6,a4,d2,c2,e2,g4,f4,d4,d6,d7,e8,h7,g7,g8,e5,e6,h6,e7,c3,c6,e1', 
                   'h6,h5,g5,f4,f5,g4,h3,h2,d4,h8,c3,b6,h4,e2,f2,d2,b4,c4,e4,e6,e7,d8,a7,b7,b8,d5,d6,a6,d7,f3,f6,d1', 
                   'a3,a4,b4,c5,c4,b5,a6,a7,e5,a1,f6,g3,a5,d7,c7,e7,g5,f5,d5,d3,d2,e1,h2,g2,g1,e4,e3,h3,e2,c6,c3,e8', 
                   'h3,h4,g4,f5,f4,g5,h6,h7,d5,h1,c6,b3,h5,e7,f7,d7,b5,c5,e5,e3,e2,d1,a2,b2,b1,d4,d3,a3,d2,f6,f3,d8', 
                   'c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4', 
                   'a6,a5,b5,c4,c5,b4,a3,a2,e4,a8,f3,g6,a4,d2,c2,e2,g4,f4,d4,d6,d7,e8,h7,g7,g8,e5,e6,h6,e7,c3,c6,e1', 
                   'h6,h5,g5,f4,f5,g4,h3,h2,d4,h8,c3,b6,h4,e2,f2,d2,b4,c4,e4,e6,e7,d8,a7,b7,b8,d5,d6,a6,d7,f3,f6,d1', 
                   'a3,a4,b4,c5,c4,b5,a6,a7,e5,a1,f6,g3,a5,d7,c7,e7,g5,f5,d5,d3,d2,e1,h2,g2,g1,e4,e3,h3,e2,c6,c3,e8', 
                   'h3,h4,g4,f5,f4,g5,h6,h7,d5,h1,c6,b3,h5,e7,f7,d7,b5,c5,e5,e3,e2,d1,a2,b2,b1,d4,d3,a3,d2,f6,f3,d8', 
                   'c1,d1,d2,e3,d3,e2,f1,g1,e5,a1,f6,c7,e1,g4,g3,g5,e7,e6,e4,c4,b4,a5,b8,b7,a7,d5,c5,c8,b5,f3,c3,h5', 
                   'f1,e1,e2,d3,e3,d2,c1,b1,d5,h1,c6,f7,d1,b4,b3,b5,d7,d6,d4,f4,g4,h5,g8,g7,h7,e5,f5,f8,g5,c3,f3,a5', 
                   'c8,d8,d7,e6,d6,e7,f8,g8,e4,a8,f3,c2,e8,g5,g6,g4,e2,e3,e5,c5,b5,a4,b1,b2,a2,d4,c4,c1,b4,f6,c6,h4', 
                   'f8,e8,e7,d6,e6,d7,c8,b8,d4,h8,c3,f2,d8,b5,b6,b4,d2,d3,d5,f5,g5,h4,g1,g2,h2,e4,f4,f1,g4,c6,f6,a4'])
    
    # player = ADP_Pre_Player(**game_kwargs)
    
    # selfplay_data = collect_selfplay_data(player, game_kwargs, n_games=3)
    # buffer.extend(selfplay_data)

    # selfplay_data2 = collect_selfplay_data(player, game_kwargs, n_games=15)
    # buffer.extend(selfplay_data2)
    
    # for history in regenerate_games(buffer, game_kwargs, start=15):
    #     print(*history)
    #     break
