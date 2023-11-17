import numpy as np
import random

import numpy as np
import random

class Gomoku:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if not hasattr(self, "board"):
            self.board = np.zeros((self.M, self.N), dtype=np.int8)
            self.adjacents = set()
            self.history = []
            self.player = self.FIRST_PLAYER
            self.winner = 0
        else:
            self.board = np.array(self.board, dtype=np.int8)
            self.adjacents = set(self.adjacents)
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

    def actions(self, only_adjacents: bool = False) -> list[tuple[int, int]]:
        moves = [
            (x, y) 
            for x in range(self.M) 
            for y in range(self.N)
            if self.is_legal((x, y))
        ]
        if only_adjacents and self.ADJ:
            move_set = set(["{},{}".format(i, j) for i, j in moves])
            move_set = self.adjacents.intersection(move_set)
            moves = [tuple(map(int, move.split(","))) for move in move_set]
        random.shuffle(moves)
        return moves
        
    def fin(self) -> bool:
        return self.winner or self.no_move()

    def no_move(self) -> bool:
        return len(self.history) == self.M * self.N

    def step(self, move: tuple[int, int]):
        x, y = move
        self.board[x, y] = self.player
        if self.ADJ:
            for dx in range(-self.ADJ, self.ADJ + 1):
                for dy in range(-self.ADJ, self.ADJ + 1):
                    new_x, new_y = x + dx, y + dy
                    if not (0 <= new_x < self.M and 0 <= new_y < self.N):
                        continue
                    self.adjacents.add("{},{}".format(new_x, new_y))
        self.history += [move]
        
    def score(self) -> float:
        if self.winner:
            return self.FIRST_PLAYER * self.winner
        return 0
        
    def is_legal(self, move: tuple[int, int]) -> bool:
        x, y = move
        return 0 <= x < self.M and 0 <= y < self.N and self.board[x, y] == 0

    def is_win(self, position: tuple[int, int]) -> bool:
        directions = [(0, 1), (1, 1), (1, 0), (1, -1)]
        for direction in directions:
            if self.is_win_line(position, direction):
                return True
        return False
    
    def find_near(self, position: tuple[int, int], values: list[int]) -> int:
        counter = 0
        directions = [(0, 1), (1, 1), (1, 0), (1, -1)]
        for direction in directions:
            counter += self.find_near_line(position, direction, values)
        return counter

    def find_near_line(self, position: tuple[int, int], direction: tuple[int, int], pattern: list[int]) -> int:
        counter = 0
        x, y = position
        dx, dy = direction
        
        for i in range(1 - len(pattern), len(pattern)):
            new_position = x + i * dx, y + i * dy 
            _, board_items = self.get_line(new_position, direction, len(pattern))
        
            if not len(board_items):
                continue
                        
            matches = True
            for bi, pi in zip(board_items, pattern):
                if bi != pi:
                    matches = False
                    break
            if matches:
                counter += 1
                
        return counter
    
    def get_line(self, position: tuple[int, int], direction: tuple[int, int], length: int) -> tuple[list[tuple[int, int]], list[int]]:
        x, y = position
        dx, dy = direction
        indices, values = [], []
        for i in range(length):
            new_x, new_y = x + i * dx, y + i * dy
            if not (0 <= new_x < self.M and 0 <= new_y < self.N):
                return [], []
            indices += [(new_x, new_y)]
            values += [self.board[new_x, new_y]]
        return indices, values

    def is_win_line(self, position: tuple[int, int], direction: tuple[int, int]) -> bool:
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
        output = ""
        if not self.winner:
            output += "Current player: " + str(self.player)
        else:
            output += "Winner: " + str(self.winner)

        output += "\n"
        for i in range(self.M):
            row = []
            for j in range(self.N):
                if self.board[i, j] == 1:
                    row += ["X"]
                elif self.board[i, j] == -1:
                    row += ["O"]
                else:
                    row += ["."]
            output += " ".join(row) + "\n"
        print_fn(output)
    