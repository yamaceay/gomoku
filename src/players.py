from .gomoku import Gomoku
from .patterns import sortfn
import numpy as np


class Player:
    def rewards_actions(self, _: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        raise NotImplementedError
    
    def next_move_probs(self, game: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        rewards_actions = self.rewards_actions(game)
        if game.player == -1:
            rewards_actions = [(-r, a) for r, a in rewards_actions[::-1]]
        
        rewards, actions = zip(*rewards_actions)
        rewards = np.array(rewards)
        probs = np.exp(rewards)
        probs /= probs.sum()
        return list(zip(probs, actions))
    
    def next_move(self, game: Gomoku, epsilon: float = 0.) -> tuple[int, int]:
        probs_actions = self.next_move_probs(game)
        # print([f"{a}: {p:.2f}" for p, a in probs_actions])
        if np.random.random() >= epsilon:
            return probs_actions[0][1]
        
        probs, actions = zip(*probs_actions)
        action_i = np.random.choice(len(actions), p=probs)
        return actions[action_i]

class RandomPlayer(Player):
    def rewards_actions(self, game: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        moves = game.actions()
        probs = np.random.random(len(moves))
        probs /= probs.sum()
        rewards_actions = [(probs[i], moves[i]) for i in range(len(moves))]
        rewards_actions = sortfn(rewards_actions, lambda x: x[0])
        return rewards_actions