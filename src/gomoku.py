import numpy as np
import random
import copy
from .patterns import PB_DICT
import re
# class GomokuSlow:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
#         if not hasattr(self, "FIRST_PLAYER"):
#             self.FIRST_PLAYER = 1
#         if not hasattr(self, "ADJ"):
#             self.ADJ = 0
#         if not hasattr(self, "board"):
#             self.board = np.zeros((self.M, self.N), dtype=np.int8)
#             self.line_cache = {}
#             self.adjacents = set()
#             self.history = []
#             self.player = self.FIRST_PLAYER
#             self.winner = 0
#         else:
#             self.board = np.array(self.board, dtype=np.int8)
#             self.line_cache = {}
#             self.adjacents = set(self.adjacents)
#             self.history = [tuple(move) for move in self.history]
    
#     def copy_state(self):
#         new_game = self.copy()
#         new_game.history = []
#         new_game.FIRST_PLAYER = new_game.player
#         return new_game
    
#     def copy(self):
#         return GomokuSlow(**self.__dict__)
    
#     def play(self, *moves: tuple[int, int]) -> tuple[float, bool]:
#         if not len(moves):
#             raise Exception("No moves provided")
        
#         for move in moves:
#             if not self.is_legal(move):
#                 raise Exception("Illegal move: " + str(move))
            
#             self.step(move)
#             if self.is_win(move):
#                 self.winner = self.player
#                 self.player = -self.player
#                 return self.score(), True
            
#             self.player = -self.player
#         return 0, self.no_move()

#     def reset(self):
#         self.history = []
#         self.player = self.FIRST_PLAYER

#     def actions(self, only_adjacents: bool = False) -> list[tuple[int, int]]:
#         moves = [
#             (x, y) 
#             for x in range(self.M) 
#             for y in range(self.N)
#             if self.is_legal((x, y))
#         ]
#         if only_adjacents and self.ADJ:
#             move_set = set(["{},{}".format(i, j) for i, j in moves])
#             move_set = self.adjacents.intersection(move_set)
#             moves = [tuple(map(int, move.split(","))) for move in move_set]
#         random.shuffle(moves)
#         return moves
        
#     def fin(self) -> bool:
#         return self.winner or self.no_move()

#     def no_move(self) -> bool:
#         return np.prod(self.board) != 0

#     def step(self, move: tuple[int, int]):
#         x, y = move
#         self.board[x, y] = self.player
#         self.line_cache = {}
#         if self.ADJ:
#             for dx in range(-self.ADJ, self.ADJ + 1):
#                 for dy in range(-self.ADJ, self.ADJ + 1):
#                     new_x, new_y = x + dx, y + dy
#                     if not (0 <= new_x < self.M and 0 <= new_y < self.N):
#                         continue
#                     self.adjacents.add("{},{}".format(new_x, new_y))
#         self.history += [move]
    
#     def score(self) -> float:
#         return self.winner * self.FIRST_PLAYER
        
#     def is_legal(self, move: tuple[int, int]) -> bool:
#         x, y = move
#         return 0 <= x < self.M and 0 <= y < self.N and self.board[x, y] == 0

#     @property
#     def directions(self) -> list[tuple[int, int]]:
#         return [(0, 1), (1, 1), (1, 0), (1, -1)]

#     def is_win(self, position: tuple[int, int]) -> bool:
#         for direction in self.directions:
#             if self.is_win_line(position, direction):
#                 return True
#         return False
    
#     def find_near(self, position: tuple[int, int], values: list[int]) -> int:
#         counter = 0
#         for direction in self.directions:
#             counter += self.find_near_line(position, direction, values)
#         return counter

#     def find_near_line(self, position: tuple[int, int], direction: tuple[int, int], pattern: list[int]) -> int:
#         counter = 0
#         x, y = position
#         dx, dy = direction
        
#         for i in range(1 - len(pattern), len(pattern)):
#             new_position = x + i * dx, y + i * dy 
#             board_items = self.get_line(new_position, direction, len(pattern))
        
#             if not len(board_items):
#                 continue
                        
#             matches = True
#             for bi, pi in zip(board_items, pattern):
#                 if bi != pi:
#                     matches = False
#                     break
#             if matches:
#                 counter += 1
                
#         return counter
    
#     def get_line(self, position: tuple[int, int], direction: tuple[int, int], length: int) -> list[int]:
#         key = (position, direction, length)
#         if key in self.line_cache:
#             return self.line_cache[key]
        
#         x, y = position
#         dx, dy = direction
#         values = []
#         for i in range(length):
#             new_x, new_y = x + i * dx, y + i * dy
#             if not (0 <= new_x < self.M and 0 <= new_y < self.N):
#                 return []
#             values += [self.board[new_x, new_y]]
            
#         self.line_cache[key] = values
#         return values

#     def is_win_line(self, position: tuple[int, int], direction: tuple[int, int]) -> bool:
#         counter = 0
        
#         x, y = position
#         dx, dy = direction
#         values = [
#             self.board[x + i * dx, y + i * dy]
#             for i in range(1 - self.K, self.K)
#             if 0 <= x + i * dx < self.M 
#             and 0 <= y + i * dy < self.N
#         ]

#         for value in values:
#             if value == self.board[x, y]:
#                 counter += 1
#                 if counter >= self.K:
#                     return True
#             else:
#                 counter = 0
#         return False

#     def print(self, print_fn = print):
#         output = ""
#         if not self.winner:
#             output += "Current player: " + str(self.player)
#         else:
#             output += "Winner: " + str(self.winner)

#         output += "\n"
#         for i in range(self.M):
#             row = []
#             for j in range(self.N):
#                 if self.board[i, j] == 1:
#                     row += ["X"]
#                 elif self.board[i, j] == -1:
#                     row += ["O"]
#                 else:
#                     row += ["."]
#             output += " ".join(row) + "\n"
#         print_fn(output)
    
dtl = lambda x: 'x' if x == 1 else '-' if x == 0 else 'o'
ltd = lambda x: 1 if x == 'x' else 0 if x == '-' else -1
def dir_to_loc(*val: int):
    return "".join(map(dtl, val))
def loc_to_dir(loc: str):
    return tuple(map(ltd, list(loc)))
def move_to_loc(*moves: tuple[int, int]):
    move_strs = []
    for move in moves:
        move_strs += [f"{chr(ord('a') + move[0])}{move[1] + 1}"]
    return ",".join(move_strs)
def loc_to_move(locs: str):
    moves = []
    for loc in locs.split(","):
        moves += [(ord(loc[0]) - ord('a'), int(loc[1:]) - 1)]
    if len(moves) == 1:
        return moves[0]
    return tuple(moves)
class Gomoku:
    def __init__(self, **kwargs):
        self.M = kwargs.pop("M")
        self.N = kwargs.pop("N")
        self.K = kwargs.pop("K")
        self.FIRST_PLAYER = kwargs.pop("FIRST_PLAYER", 1)
        self.ADJ = kwargs.pop("ADJ", 0)
        self.__dict__.update(kwargs)
        if "board" not in kwargs:
            self.board = np.zeros((self.M, self.N), dtype=np.int8)
            self.line_cache = {len(pattern): {} for pattern in PB_DICT}
            self.adjacents = set()
            self.history = ""
            self.player = self.FIRST_PLAYER
            self.winner = 0
        else:
            self.board = copy.deepcopy(self.board)
            self.line_cache = copy.deepcopy(self.line_cache)
            self.adjacents = copy.deepcopy(self.adjacents)
    
    def copy_state(self):
        new_game = self.copy()
        new_game.history = ""
        new_game.FIRST_PLAYER = new_game.player
        return new_game
    
    def copy(self):
        return Gomoku(**self.__dict__)
    
    def play(self, *moves: tuple[int, int]) -> tuple[float, bool]:
        if not len(moves):
            raise Exception("No moves provided")
        
        for move in moves:
            if not self.is_legal(move):
                raise Exception("Illegal move: " + str(move))
            
            self.step(move)
            if self.is_win(move):
                self.winner = self.player
                self.player = -self.player
                return self.score(), True
            
            self.player = -self.player
        return 0, self.no_move()

    def reset(self):
        self.history = ""
        self.player = self.FIRST_PLAYER

    def actions(self, only_adjacents: bool = False) -> list[tuple[int, int]]:
        moves = [
            (x, y) 
            for x in range(self.M) 
            for y in range(self.N)
            if self.is_legal((x, y))
        ]
        if only_adjacents and self.ADJ:
            move_set = set(move_to_loc(move) for move in moves)
            move_set = self.adjacents.intersection(move_set)
            if not len(move_set):
                return []
            moves = list(loc_to_move(",".join(move_set)))
        random.shuffle(moves)
        return moves
        
    def fin(self) -> bool:
        return self.winner or self.no_move()

    def no_move(self) -> bool:
        return np.prod(self.board) != 0

    def step(self, move: tuple[int, int]):
        x, y = move
        self.board[x, y] = self.player
        lengths = set(map(len, PB_DICT))
        for dx, dy in self.directions:
            for length in sorted(lengths):
                for i in range(1 - length, length):
                    new_x, new_y = x + i * dx, y + i * dy
                    if not (0 <= new_x < self.M and 0 <= new_y < self.N):
                        continue
                    self.get_line((new_x, new_y), (dx, dy), length)
        if self.ADJ:
            for (dx, dy) in self.directions:
                for i in range(-self.ADJ, self.ADJ + 1):
                    new_x, new_y = x + i * dx, y + i * dy
                    if not (0 <= new_x < self.M and 0 <= new_y < self.N):
                        continue
                    self.adjacents.add(move_to_loc((new_x, new_y)))
        if not len(self.history):
            self.history = move_to_loc(move)
        else:
            self.history += "," + move_to_loc(move)
    
    def score(self) -> float:
        return self.winner * self.FIRST_PLAYER
        
    def is_legal(self, move: tuple[int, int]) -> bool:
        x, y = move
        return 0 <= x < self.M and 0 <= y < self.N and self.board[x, y] == 0

    @property
    def directions(self) -> list[tuple[int, int]]:
        return [
            (0, 1), (1, 1), (1, 0), (1, -1), 
            # (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]

    def is_win(self, position: tuple[int, int]) -> bool:
        for direction in self.directions:
            if self.is_win_line(position, direction):
                return True
        return False
    
    def update_line_cache(self, 
                          length: int, 
                          position: tuple[int, int], 
                          direction: tuple[int, int], 
                          indices: str, 
                          values: str):
        position_loc = move_to_loc(position)
        direction_loc = dir_to_loc(*direction)
        if position_loc not in self.line_cache[length]:
            self.line_cache[length][position_loc] = {}
        self.line_cache[length][position_loc][direction_loc] = [indices, values]
    
    def get_line_cache(self,
                       length: int = None,
                       position: tuple[int, int] = None,
                       direction: tuple[int, int] = None):
        if length is None:
            return list(self.line_cache)
        elif position is None:
            return list(map(loc_to_move, self.line_cache[length]))
        else:
            position_loc = move_to_loc(position)
            if direction is None:
                return list(map(loc_to_dir, self.line_cache[length][position_loc]))
            else:
                direction_loc = dir_to_loc(*direction)
                if position_loc in self.line_cache[length]:
                    if direction_loc in self.line_cache[length][position_loc]:
                        indices_loc, values_loc = self.line_cache[length][position_loc][direction_loc]
                        return indices_loc, values_loc
        return None
    
    def get_line(self, position: tuple[int, int], direction: tuple[int, int], length: int) -> tuple[list[tuple[int, int]], list[int]]:
        # cached_line = self.get_line_cache(length, position, direction)
        # if cached_line is not None:
        #     return loc_to_move(cached_line[0]), loc_to_dir(cached_line[1])
        
        x, y = position
        dx, dy = direction
        indices, values = [], []
        for i in range(length):
            new_x, new_y = x + i * dx, y + i * dy
            if not (0 <= new_x < self.M and 0 <= new_y < self.N):
                return [], []
            indices += [(new_x, new_y)]
            values += [self.board[new_x, new_y]]
            
        if len(values):
            indices_loc, values_loc = move_to_loc(*indices), dir_to_loc(*values)
            self.update_line_cache(length, position, direction, indices_loc, values_loc)
        return indices, values

    def is_win_line(self, position: tuple[int, int], direction: tuple[int, int]) -> bool:
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
    
if __name__ == "__main__":
    from .adp import ADP_Player
    game_kwargs = {
        'M': 10,
        'N': 10,
        'K': 5,
        'ADJ': 2,
    }
    
    value_network_kwargs = {
        'alpha': 0.9,
        'magnify': 2,
        'gamma': 0.9,
        'lr': 0.01,
        'n_steps': 1, 
    }
    
    policy_network_kwargs = {
        'epsilon': 0.1,
    }
    
    player = ADP_Player("models_wzlen/best.h5", value_network_kwargs, policy_network_kwargs)
    game = Gomoku(**game_kwargs)
    while not game.fin():
        action = player.next_move(game)
        value_list, _ = player.value_network.extract_values(game)
        for values in value_list.values():
            found_values = [value for value in values if value in PB_DICT]
            print(found_values)
        # print(player.value_network.extract_features(game, value_list))
        game.play(action)
        game.print()