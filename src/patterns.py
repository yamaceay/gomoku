from typing import Callable
DECAY = 0.9
BASIS = 10

class Pattern:
    @staticmethod
    def dir_to_loc(*val: int) -> str:
        return "".join(map(Pattern._itl, val))
    
    @staticmethod
    def loc_to_dir(loc: str) -> tuple[int, int]:
        return tuple(map(Pattern._lti, list(loc)))
    
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
    
    @staticmethod
    def revp(pattern: str) -> str: 
        return ''.join([Pattern._itl(-Pattern._lti(c)) for c in pattern])
    
    @staticmethod
    def _lti(c: str) -> int: 
        return 1 if c == 'x' else -1 if c == 'o' else 0
    
    @staticmethod
    def _itl(c: str) -> str: 
        return 'x' if c == 1 else 'o' if c == -1 else '-'

PB_DICT_5 = {
    '-oooo-': (4, 0),
    'x-ooo-x': (3, 3),
    'x-ooo--': (3, 1),
    '--ooo--': (3, 0),
    '--ooo-o': (3, 0),
    'o-ooo-o': (4, 0),
    'x-ooo-o': (3, 1),
    '--oo--': (2, 0),
    'x-oo--': (2, 1),
    'xoo-o--': (2, 2),
    'xo-oo--': (2, 2),
    'xo-oo-x': (2, 3),
    'xoo-o-x': (2, 3),
    'xooo--':  (2, 0),
    'xoo---': (0, 0),
    'xoooo-': (3, 0),
    '-oo-o-': (3, 1),
    'xooo-o-': (3, 1),
    'xoo-oo-': (3, 1),
    'xooo-ox': (3, 3),
    'xoo-oox': (3, 3),
    'xooo-oo': (3, 1),
    '-o-o-o-': (3, 2),
    'xo-o-ox': (3, 6),
    'xo-o-o-': (3, 4),
    '--o-o--': (2, 2),
    'x-o-o--': (2, 4),
    'x-o-o-x': (2, 6),
    '--o--': (0, 0),
}

def pb_heuristic(pb: tuple[int, int]) -> float:
    return BASIS ** (pb[0] - 5) * DECAY ** pb[1]

def sortfn(items: list, key: Callable = None, reverse: bool = True) -> list:
    sorted_args = {}
    if key is None:
        sorted_args.update(dict(key=key))
    sorted_list = sorted(items, **sorted_args)
    if reverse:
        sorted_list = reversed(sorted_list)
    return list(sorted_list)

# def calculate_exponents(pattern: str) -> tuple[int, int]:
#     # Count the number of 'o' characters
#     basis_exp = pattern.count('o')

#     # Find the position of the first and last 'o' or 'x' character
#     first_char = min(i if (i := pattern.find('o')) != -1 else len(pattern), 
#                      i if (i := pattern.find('x')) != -1 else len(pattern))
#     last_char = max(i if (i := pattern.rfind('o')) != -1 else -1, 
#                      i if (i := pattern.rfind('x')) != -1 else -1)

#     # Calculate decay_exp based on the number of '-' characters before the first 'o' or 'x' and after the last 'o' or 'x'
#     decay_exp = pattern[:first_char].count('-') + pattern[last_char+1:].count('-')

#     return basis_exp, decay_exp

# if __name__ == "__main__":
#     for x, y in PB_DICT_5.items():
#         fx = calculate_exponents(x)
#         if y[0] != fx[0]:
#             print(f"Input: {x}, Type: Basis, Expected: {y[0]}, Output: {fx[0]}")
#         if y[1] != fx[1]:
#             print(f"Input: {x}, Type: Decay, Expected: {y[1]}, Output: {fx[1]}")
    
    # import numpy as np
    # features, labels = [], []
    # for pattern, y in PB_DICT_5.items():
    #     xs = list(map(Pattern._lti, pattern))
    #     if len(xs) < 7:
    #         xs += [1] * (7 - len(xs))
    #     features += [xs]
    #     labels += [list(y)]
        
    # features = np.array(features)
    # labels = np.array(labels)
    
    # from sklearn.neural_network import MLPClassifier
    # classifiers = [MLPClassifier(max_iter=10000) for _ in range(2)]
    # for i, classifier in enumerate(classifiers):
    #     classifier.fit(features, labels[:, i])

    # avg_true = 0
    # for feature, label in zip(features, labels):
    #     pred = np.array([classifier.predict([feature])[0] for classifier in classifiers])
    #     if np.all(pred == label):
    #         avg_true += 1
    #     else:
    #         print(f"Predicted: {pred}, Actual: {label}")
    
    # print(f"Accuracy: {avg_true / len(features)}")
    # for classifier in classifiers:
    #     print(classifier.coefs_)
    #     print(classifier.intercepts_)
    #     print()