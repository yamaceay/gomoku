import numpy as np
import random

class Gomoku:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if not hasattr(self, "board"):
            self.board = np.zeros((self.M, self.N), dtype=np.int8)
            self.history = []
            self.player = self.FIRST_PLAYER
            self.winner = 0
        else:
            self.board = np.array(self.board, dtype=np.int8)
            self.history = [tuple(move) for move in self.history]
    
    def copy(self, include_history=False):
        new_game = Gomoku(**self.__dict__)
        if not include_history:
            new_game.history = []
            new_game.FIRST_PLAYER = new_game.player
        return new_game
    
    def play(self, *moves: tuple[int, int]) -> tuple[float, bool]:
        if not len(moves):
            raise Exception("No moves provided")
        
        for move in moves:
            if not self.is_legal(move):
                raise Exception("Illegal move: " + str(move))
            
            self.step(move)
            if self.is_win(move):
                self.winner = self.player
                return self.score(), True
            
            self.player = -self.player
        return 0, self.no_move()

    def reset(self):
        self.history = []
        self.player = self.FIRST_PLAYER

    def actions(self):
        moves = [
            (x, y) 
            for x in range(self.M) 
            for y in range(self.N)
            if self.is_legal((x, y))
        ]
        random.shuffle(moves)
        return moves
        
    def fin(self):
        return self.winner or self.no_move()

    def no_move(self):
        return len(self.history) == self.M * self.N

    def step(self, move: tuple[int, int]):
        x, y = move
        self.board[x, y] = self.player
        self.history += [move]
        
    def score(self):
        if self.winner:
            return self.FIRST_PLAYER * self.winner
        return 0
        
    def is_legal(self, move: tuple[int, int]):
        x, y = move
        return 0 <= x < self.M and 0 <= y < self.N and self.board[x, y] == 0

    def is_win(self, position: tuple[int, int]):
        directions = [(0, 1), (1, 1), (1, 0), (1, -1)]
        for direction in directions:
            if self.is_win_line(position, direction):
                return True
        return False
    
    def is_win_line(self, position: tuple[int, int], direction: tuple[int, int]):
        x, y = position
        dx, dy = direction
        
        counter = 0        
        for i in range(1 - self.K, self.K):
            if counter >= self.K:
                break
            if i != 0:
                new_x, new_y = x + i * dx, y + i * dy
                if not (0 <= new_x < self.M and 0 <= new_y < self.N):
                    continue
                if self.board[new_x, new_y] != self.board[x, y]:
                    counter = 0
                    continue
            counter += 1
        return counter >= self.K

    def print(self, print_fn = print):
        if not self.winner:
            print_fn("Current player: " + str(self.player))
        else:
            print_fn("Winner: " + str(self.winner))

        for i in range(self.M):
            row = []
            for j in range(self.N):
                if self.board[i, j] == 1:
                    row += ["X"]
                elif self.board[i, j] == -1:
                    row += ["O"]
                else:
                    row += ["."]
            print_fn(" ".join(row))
    