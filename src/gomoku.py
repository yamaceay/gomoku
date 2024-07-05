from typing import Callable
import numpy as np

import random
import copy

def sortfn(items: list, key: Callable = None, reverse: bool = True) -> list:
    sorted_args = {}
    if key is None:
        sorted_args.update(dict(key=key))
    sorted_list = sorted(items, **sorted_args)
    if reverse:
        sorted_list = reversed(sorted_list)
    return list(sorted_list)

SMALL_GAME = S_GAME = (6, 6, 4)
MEDIUM_GAME = M_GAME = (8, 8, 5)
LARGE_GAME = L_GAME = (10, 10, 5)

class Gomoku:
    def __init__(self, M: int, N: int, K: int):
        assert M > 0 and N > 0 and K > 0, "Invalid game parameters: {}, {}, {}".format(M, N, K)
        self.M = M
        self.N = N
        self.K = K
        
        self.player = 1
        self.board = np.zeros((self.M, self.N), dtype=np.int8)
        self.last_move = None
        self.history = []
        
        self._winner = 0
        self._legal_actions = set([(x, y) for x in range(self.M) for y in range(self.N)])
        self._directions = [
            (0, 1), 
            (1, 1), 
            (1, 0), 
            (1, -1)
        ]
    
    def copy(self):
        return copy.deepcopy(self)
    
    def play(self, *moves: tuple[int, int]) -> tuple[float, bool]:
        if not len(moves):
            raise Exception("No moves provided")
        
        for move in moves:
            if not self._is_legal(move):
                raise Exception("Illegal move: " + str(move))
            
            self._step(move)
            if self._is_win(move):
                self._winner = self.player
                self.player = -self.player
                return self.score(), True
            
            self.player = -self.player
        return 0, self.no_move()

    def actions(self) -> list[tuple[int, int]]:
        moves = list(self._legal_actions)
        random.shuffle(moves)
        return moves
        
    def fin(self) -> bool:
        return self.score() or self.no_move()
    
    def score(self) -> float:
        return self._winner

    def no_move(self) -> bool:
        return len(self._legal_actions) == 0
    
    def encode(self) -> np.ndarray:
        states = np.zeros((4, self.M, self.N), dtype=np.float32)
        states[0] = np.asarray(self.board == 1, dtype=np.float32)
        states[1] = np.asarray(self.board == -1, dtype=np.float32)
        if self.last_move is not None:
            states[2][self.last_move] = 1.
        if self.player == 1:
            states[3] = 1.
        return states[:, ::-1, :]

    def _step(self, move: tuple[int, int]) -> None:
        x, y = move
        self.board[x, y] = self.player
        self._legal_actions.remove(move)
        self.last_move = move
        self.history += [move]
        
    def _is_legal(self, move: tuple[int, int]) -> bool:
        assert isinstance(move, (tuple, list)), "Move must be a tuple of integers, got: {}".format(move)
        x, y = move
        return 0 <= x < self.M and 0 <= y < self.N and self.board[x, y] == 0

    def _is_win(self, position: tuple[int, int]) -> bool:
        for direction in self._directions:
            if self._is_win_line(position, direction):
                return True
        return False

    def _is_win_line(self, position: tuple[int, int], direction: tuple[int, int]) -> bool:
        counter = 0
        x, y = position
        dx, dy = direction
        values = [
            self.board[x + i * dx, y + i * dy]
            for i in range(1 - self.K, self.K)
            if 0 <= x + i * dx < self.M 
            and 0 <= y + i * dy < self.N
        ]

        for value in values:
            if value == self.board[x, y]:
                counter += 1
                if counter >= self.K:
                    return True
            else:
                counter = 0
        return False
    
    def __repr__(self) -> str:
        output = ""
        if not self.score():
            if self.last_move is not None:
                player_str = "X" if self.player == -1 else "O"
                (i, j) = self.last_move
                last_move_str = Pattern.move_to_loc((i, j - 1))
                output += f"Last move: {player_str} in {last_move_str}"
        elif self.no_move():
            output += "Tie break"
        else:
            winner = self.score()
            winner_str = "X" if winner == 1 else "O"
            output += f"Winner: {winner_str}"

        output += "\n"
        output += " " + " ".join([""] + [str(i) for i in range(self.N)]) + "\n"
        for i in range(self.M):
            row = [chr(ord('a') + i)]
            for j in range(self.N):
                if self.board[i, j] == 1:
                    row += ["X"]
                elif self.board[i, j] == -1:
                    row += ["O"]
                else:
                    row += ["."]
            output += " ".join(row) + "\n"
            
        return output
class Pattern:
    @staticmethod
    def move_to_loc(*moves: tuple[int, int]) -> str:
        assert len(moves), "No move given"
        move_strs = []
        for move in moves:
            x_str = chr(ord('a') + move[0])
            y_str = move[1] + 1
            move_str = f"{x_str}{y_str}"
            move_strs += [move_str]
        return ",".join(move_strs)
    
    @staticmethod
    def loc_to_move_one(loc: str) -> tuple[int, int]:
        moves = Pattern.loc_to_move(loc)
        assert len(moves) == 1, "More than one move given"
        return moves[0]

    @staticmethod
    def loc_to_move(locs: str) -> tuple[tuple[int, int], ...]:
        assert len(locs), "No string given"
        moves = []
        for loc in locs.split(","):
            x = ord(loc[0]) - ord('a')
            y = int(loc[1:]) - 1
            move = (x, y)
            moves += [move]
        return tuple(moves)