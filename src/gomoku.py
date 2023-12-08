import numpy as np
import random
import copy
from .patterns import PB_DICT, revp, move_to_loc, loc_to_move, loc_to_move_one, dir_to_loc, loc_to_dir

class Gomoku:
    def __init__(self, **kwargs):
        self.M = kwargs.pop("M")
        self.N = kwargs.pop("N")
        self.K = kwargs.pop("K")
        self.ADJ = kwargs.pop("ADJ", 0)
        self.__dict__.update(kwargs)
        if "board" not in kwargs:
            self.board = np.zeros((self.M, self.N), dtype=np.int8)
            self.line_cache = {len(pattern): {} for pattern in PB_DICT}
            self.adjacents = set()
            self._history = ""
            self.player = 1
            self.winner = 0
        else:
            self.board = copy.deepcopy(self.board)
            self.line_cache = copy.deepcopy(self.line_cache)
            self.adjacents = copy.deepcopy(self.adjacents)
    
    def copy(self):
        return Gomoku(**self.__dict__)
    
    def play(self, *moves: tuple[int, int]) -> tuple[float, bool]:
        if not len(moves):
            raise Exception("No moves provided")
        
        for move in moves:
            if not self._is_legal(move):
                raise Exception("Illegal move: " + str(move))
            
            self._step(move)
            if self._is_win(move):
                self.winner = self.player
                self.player = -self.player
                return self.score(), True
            
            self.player = -self.player
        return 0, self.no_move()

    def actions(self) -> list[tuple[int, int]]:
        moves = [
            (x, y) 
            for x in range(self.M) 
            for y in range(self.N)
            if self._is_legal((x, y))
        ]
        if self.ADJ and len(self._history):
            move_set = set(move_to_loc(move) for move in moves)
            move_set = self.adjacents.intersection(move_set)
            if not len(move_set):
                return []
            moves = list(loc_to_move(",".join(move_set)))
        random.shuffle(moves)
        return moves
        
    def fin(self) -> bool:
        return self.winner or self.no_move()
    
    def score(self) -> float:
        return self.winner

    def no_move(self) -> bool:
        return np.where(self.board == 0, 1, 0).sum() == 0
        
    @property
    def last_move(self):
        history = self.history()
        if not len(history):
            return None
        return history[-1]

    def history(self, rot: bool = False, lrf: bool = False, udf: bool = False) -> list[tuple[int, int]]:
        if not len(self._history):
            return []
        moves = list(loc_to_move(self._history))
        transform_fn = lambda move: self._move_forward(move, rot, lrf, udf)
        return list(map(transform_fn, moves))
    
    def find_patterns(self, move: tuple[int, int]) -> float:
        value_list = {len(pattern): [] for pattern in PB_DICT}
        for length in self.get_line_cache():
            for direction in self.get_line_cache(length, move):
                _, values = self.get_line_cache(length, move, direction)
                value_list[length] += [values]
        
        score_list = {}
        for pattern in PB_DICT:
            pattern_o = pattern
            pattern_x = revp(pattern_o)
            for value in value_list[len(pattern)]:
                len_diff = len(pattern) - len(value)
                assert 0 <= len_diff <= 1, "Length difference of pattern vs. line: {}".format(len_diff)
                add_bound = len_diff > 0
                if value == pattern_x[:len(value)]:
                    if not add_bound or pattern_x[len(value)] == 'o':
                        if pattern not in score_list:
                            score_list[pattern] = [0, 0]
                        score_list[pattern][0] += 1
                if value == pattern_o[:len(value)]:
                    if not add_bound or pattern_o[len(value)] == 'x':
                        if pattern not in score_list:
                            score_list[pattern] = [0, 0]
                        score_list[pattern][1] += 1
        return score_list
    
    def to_zero_input(self):
        size = (self.M, self.N)
        states = np.zeros((4, *size), dtype=np.float32)
        states[0] = np.asarray(self.board == 1, dtype=np.float32)
        states[1] = np.asarray(self.board == -1, dtype=np.float32)
        if self.last_move is not None:
            states[2][self.last_move] = 1.
        if self.player == 1:
            states[3] = 1.
        return states
    
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
            return list(map(loc_to_move_one, self.line_cache[length]))
        else:
            position_loc = move_to_loc(position)
            if direction is None:
                if position_loc in self.line_cache[length]:
                    return list(map(loc_to_dir, self.line_cache[length][position_loc]))
            else:
                direction_loc = dir_to_loc(*direction)
                if position_loc in self.line_cache[length]:
                    if direction_loc in self.line_cache[length][position_loc]:
                        indices_loc, values_loc = self.line_cache[length][position_loc][direction_loc]
                        return indices_loc, values_loc
        return []
        
    def _move_forward(self, move: tuple[int, int], rot: bool = False, lrf: bool = False, udf: bool = False) -> tuple[int, int]:
        x, y = move
        if rot:
            x, y = self.N - 1 - y, x
        if lrf:
            y = self.N - 1 - y
        if udf:
            x = self.M - 1 - x
        return x, y

    def _step(self, move: tuple[int, int]):
        x, y = move
        self.board[x, y] = self.player
        lengths = set(map(len, PB_DICT))
        for dx, dy in self._directions:
            for length in sorted(lengths):
                for i in range(1 - length, length):
                    new_x, new_y = x + i * dx, y + i * dy
                    if not (0 <= new_x < self.M and 0 <= new_y < self.N):
                        continue
                    position, direction = (new_x, new_y), (dx, dy)
                    indices, values = self._try_get_line(length, position, direction)
                    if not len(indices) or all([v == 0 for v in values]):
                        continue
                    indices_loc, values_loc = move_to_loc(*indices), dir_to_loc(*values)
                    self.update_line_cache(length, position, direction, indices_loc, values_loc)
        if self.ADJ:
            for (dx, dy) in self._directions:
                for i in range(-self.ADJ, self.ADJ + 1):
                    new_x, new_y = x + i * dx, y + i * dy
                    if not (0 <= new_x < self.M and 0 <= new_y < self.N):
                        continue
                    self.adjacents.add(move_to_loc((new_x, new_y)))
        
        if not len(self._history):
            self._history = move_to_loc(move)
        else:
            self._history += "," + move_to_loc(move)
        
    def _is_legal(self, move: tuple[int, int]) -> bool:
        assert isinstance(move, (tuple, list)), "Move must be a tuple of integers, got: {}".format(move)
        x, y = move
        return 0 <= x < self.M and 0 <= y < self.N and self.board[x, y] == 0

    @property
    def _directions(self) -> list[tuple[int, int]]:
        return [(0, 1), (1, 1), (1, 0), (1, -1)]

    def _is_win(self, position: tuple[int, int]) -> bool:
        for direction in self._directions:
            if self._is_win_line(position, direction):
                return True
        return False
    
    def _try_get_line(self, length: int, position: tuple[int, int], direction: tuple[int, int]) -> tuple[list[tuple[int, int]], list[int]]:
        x, y = position
        dx, dy = direction
        indices, values = [], []
        bound = False
        for i in range(length):
            new_x, new_y = x + i * dx, y + i * dy
            indices += [(new_x, new_y)]
            try:
                if bound:
                    return [], []
                values += [self.board[new_x, new_y]]
            except:
                bound = True
        return indices, values

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
    
    def __repr__(self):
        output = ""
        if not self.winner:
            output += "Current player: " + str(self.player)
        elif self.no_move():
            output += "Tie break"
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
            
        return output
    
if __name__ == "__main__":
    
    game_kwargs = {
        'M': 8,
        'N': 8,
        'K': 5,
        'ADJ': 2,
    }
    
    game = Gomoku(**game_kwargs)
    
    history = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    game.play(*history)
    print(game)
    
    rotated_history = game.history(True, True, True)
    
    new_game = Gomoku(**game_kwargs)
    new_game.play(*rotated_history)
    print(new_game)