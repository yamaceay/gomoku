import numpy as np
class Gomoku:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.board = np.zeros((self.M, self.N), dtype=np.int8)
        self.player = self.FIRST_PLAYER
        self.winner = None
    
    def print(self):
        if not self.winner:
            print("Current player:", self.player)
        else:
            print(f"Winner: {self.winner}")

        for i in range(self.M):
            row = []
            for j in range(self.N):
                if self.board[i, j] == 1:
                    row += ["X"]
                elif self.board[i, j] == -1:
                    row += ["O"]
                else:
                    row += ["."]
            print(" ".join(row))
        
    def is_legal(self, move: tuple[int, int]):
        x, y = move
        return 0 <= x < self.M and 0 <= y < self.N and self.board[x, y] == 0

    def do_move(self, move: tuple[int, int]):
        x, y = move
        self.board[x, y] = self.player

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
                    return False
                if self.board[new_x, new_y] != self.board[x, y]:
                    counter = 0
                    break
            counter += 1
        return counter >= self.K

    def is_win(self, position: tuple[int, int]):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for direction in directions:
            if self.is_win_line(position, direction):
                return True
        return False

    def play(self, moves: list[tuple[int, int]]):
        for move in moves:
            if not self.is_legal(move):
                raise Exception("Illegal move: " + str(move))
            
            self.do_move(move)
            if self.is_win(move):
                self.winner = self.player
                break
            
            self.player = -self.player

        return self.winner