import numpy as np
import random
import copy
from .patterns import PB_DICT_5, Pattern

class Gomoku:
    def __init__(self, M: int, N: int, K: int, ADJ: int = 0):
        self.M = M
        self.N = N
        self.K = K
        self.ADJ = ADJ
        self.player = 1
        
        self.play_only = False
        self.board = np.zeros((self.M, self.N), dtype=np.int8)
        self.last_move = None
        
        self._line_cache = {len(pattern): {} for pattern in PB_DICT_5}
        self._adjacents = set()
        self._history = ""
        self._winner = 0
        self._legal_actions = set([(x, y) for x in range(self.M) for y in range(self.N)])
        self._directions = [
            (0, 1), 
            (1, 1), 
            (1, 0), 
            (1, -1)
        ]
        self._transformations = [
            (False, False, False),
            (False, False, True),
            (False, True, False),
            (False, True, True),
            (True, False, False),
            (True, False, True),
            (True, True, False),
            (True, True, True)
        ]
    
    def set_play_only(self) -> None:
        self.play_only = True
    
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
        moves = self._legal_actions
        if self.ADJ and len(self._history):
            moves = self._adjacents.intersection(moves)
            if not len(moves):
                return []
        moves = list(moves)
        random.shuffle(moves)
        return moves
        
    def fin(self) -> bool:
        return self.score() or self.no_move()
    
    def score(self) -> float:
        return self._winner

    def no_move(self) -> bool:
        return len(self._legal_actions) == 0

    def history(self, rot: bool = False, lrf: bool = False, udf: bool = False) -> list[tuple[int, int]]:
        if not len(self._history):
            return []
        moves = list(Pattern.loc_to_move(self._history))
        transform_fn = lambda move: self._move_forward(move, rot, lrf, udf)
        return list(map(transform_fn, moves))
    
    def history_str(self, *args, **kwargs) -> str:
        if not len(self._history):
            return ""
        return Pattern.move_to_loc(*self.history(*args, **kwargs))
    
    def history_str_aug(self) -> list[str]:
        return [
            self.history_str(*transformation)
            for transformation in self._transformations
        ]
    
    def find_patterns(self, move: tuple[int, int] = None) -> float:
        value_list = {len(pattern): [] for pattern in PB_DICT_5}
        for length in self.get_line_cache():
            moves = [move] if move is not None else self.get_line_cache(length)
            for move in moves:
                for direction in self.get_line_cache(length, move):
                    for values in self.get_line_cache(length, move, direction):
                        value_list[length] += [values]
                        break
        
        score_list = {}
        for pattern in PB_DICT_5:
            pattern_o = pattern
            pattern_x = Pattern.revp(pattern_o)
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
    
    def to_zero_input(self) -> np.ndarray:
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
                          values: str) -> None:
        position_loc = Pattern.move_to_loc(position)
        direction_loc = Pattern.dir_to_loc(*direction)
        if position_loc not in self._line_cache[length]:
            self._line_cache[length][position_loc] = {}
        self._line_cache[length][position_loc][direction_loc] = [indices, values]
    
    def get_line_cache(self,
                       length: int = None,
                       position: tuple[int, int] = None,
                       direction: tuple[int, int] = None) -> list[tuple[int, int]]:
        if length is None:
            for l in self._line_cache:
                yield l
        elif position is None:
            for p in map(Pattern.loc_to_move_one, self._line_cache[length]):
                yield p
        else:
            position_loc = Pattern.move_to_loc(position)
            assert position_loc in self._line_cache[length], "Position {} not in line cache".format(position_loc)
            if direction is None:
                for d in map(Pattern.loc_to_dir, self._line_cache[length][position_loc]):
                    yield d
            else:
                direction_loc = Pattern.dir_to_loc(*direction)
                assert direction_loc in self._line_cache[length][position_loc], "Direction {} not in line cache".format(direction_loc)
                _, values_loc = self._line_cache[length][position_loc][direction_loc]
                yield values_loc
                return
        
    def _move_forward(self, move: tuple[int, int], rot: bool = False, lrf: bool = False, udf: bool = False) -> tuple[int, int]:
        x, y = move
        if rot:
            x, y = self.N - 1 - y, x
        if lrf:
            y = self.N - 1 - y
        if udf:
            x = self.M - 1 - x
        return x, y

    def _step(self, move: tuple[int, int]) -> None:
        x, y = move
        self.board[x, y] = self.player
        self._legal_actions.remove(move)
        self.last_move = move
        if not self.play_only:
            lengths = set(map(len, PB_DICT_5))
            for dx, dy in self._directions:
                for length in sorted(lengths):
                    for i in range(1 - length, length):
                        new_x, new_y = x - i * dx, y - i * dy
                        if not (0 <= new_x < self.M and 0 <= new_y < self.N):
                            continue
                        position, direction = (new_x, new_y), (dx, dy)
                        indices, values = self._try_get_line(length, position, direction)
                        if not len(indices) or all([v == 0 for v in values]):
                            continue
                        indices_loc, values_loc = Pattern.move_to_loc(*indices), Pattern.dir_to_loc(*values)
                        self.update_line_cache(length, position, direction, indices_loc, values_loc)
        
        # for length in self.get_line_cache():
        #     for position in self.get_line_cache(length):
        #         for direction in self.get_line_cache(length, position):
        #             for values in self.get_line_cache(length, position, direction):
        #                 indices, old_values = self._try_get_line(length, position, direction)
        #                 old_values = Pattern.dir_to_loc(*old_values)
        #                 assert old_values == values, "Line cache is corrupted at {}-{}-{}: {}, {}, {}, {}, {}".format(length, position, direction, old_values, values, self, self.last_move, Pattern.move_to_loc(*indices))
        if self.ADJ:
            for (dx, dy) in self._directions:
                for i in range(-self.ADJ, self.ADJ + 1):
                    for j in range(-self.ADJ, self.ADJ + 1):
                        new_x, new_y = x + i * dx, y + j * dy
                        if not (0 <= new_x < self.M and 0 <= new_y < self.N):
                            continue
                        self._adjacents.add((new_x, new_y))
        
        if not len(self._history):
            self._history = Pattern.move_to_loc(move)
        else:
            self._history += "," + Pattern.move_to_loc(move)
        
    def _is_legal(self, move: tuple[int, int]) -> bool:
        assert isinstance(move, (tuple, list)), "Move must be a tuple of integers, got: {}".format(move)
        x, y = move
        return 0 <= x < self.M and 0 <= y < self.N and self.board[x, y] == 0

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
            if 0 <= new_x < self.M and 0 <= new_y < self.N:
                value = self.board[new_x, new_y]
                values += [value]
            elif not bound:
                bound = True
            else:
                return [], []
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
    
    def __repr__(self) -> str:
        output = ""
        if not self.score():
            output += "Current player: " + str(self.player)
        elif self.no_move():
            output += "Tie break"
        else:
            output += "Winner: " + str(self.score())

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