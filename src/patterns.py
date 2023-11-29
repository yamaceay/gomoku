lti = lambda c: 1 if c == 'x' else -1 if c == 'o' else 0
itl = lambda c: 'x' if c == 1 else 'o' if c == -1 else '-'
revp = lambda p: ''.join([itl(-lti(c)) for c in p])

def dir_to_loc(*val: int):
    return "".join(map(itl, val))
def loc_to_dir(loc: str):
    return tuple(map(lti, list(loc)))
def move_to_loc(*moves: tuple[int, int]):
    assert len(moves), "No move given"
    move_strs = []
    for move in moves:
        x_str = chr(ord('a') + move[0])
        y_str = move[1] + 1
        move_str = f"{x_str}{y_str}"
        move_strs += [move_str]
    return ",".join(move_strs)
def loc_to_move_one(loc: str):
    moves = loc_to_move(loc)
    assert len(moves) == 1, "More than one move given"
    return moves[0]

def loc_to_move(locs: str):
    assert len(locs), "No string given"
    moves = []
    for loc in locs.split(","):
        x = ord(loc[0]) - ord('a')
        y = int(loc[1:]) - 1
        move = (x, y)
        moves += [move]
    return tuple(moves)

PB_DICT = {
    '-oooo-': lambda _: 10000,
    'x-ooo-x': lambda decay: 1000*decay**3,
    'x-ooo--': lambda decay: 1000*decay**1,
    '--ooo--': lambda _: 1000,
    '--ooo-o': lambda _: 1000,
    'o-ooo-o': lambda _: 10000,
    'x-ooo-o': lambda decay: 1000*decay**1,
    '--oo--': lambda _: 100,
    'x-oo--': lambda decay: 100*decay**1,
    'xoo-o--': lambda decay: 100*decay**2,
    'xo-oo--': lambda decay: 100*decay**2,
    'xo-oo-x': lambda decay: 100*decay**3,
    'xoo-o-x': lambda decay: 100*decay**3,
    'xooo--':  lambda _: 100,
    'xoo---': lambda _: 1,
    'xoooo-': lambda _: 1000,
    '-oo-o-': lambda decay: 1000*decay**1,
    'xooo-o-': lambda decay: 1000*decay**1,
    'xoo-oo-': lambda decay: 1000*decay**1,
    'xooo-ox': lambda decay: 1000*decay**3,
    'xoo-oox': lambda decay: 1000*decay**3,
    'xooo-oo': lambda decay: 1000*decay**1,
    '-o-o-o-': lambda decay: 1000*decay**2,
    'xo-o-ox': lambda decay: 1000*decay**6,
    'xo-o-o-': lambda decay: 1000*decay**4,
    '--o-o--': lambda decay: 100*decay**2,
    'x-o-o--': lambda decay: 100*decay**4,
    'x-o-o-x': lambda decay: 100*decay**6,
    '--o--': lambda _: 1,
}

WIN_ENCODE = [
    '-oooo',
    'o-ooo',
    'oo-oo',
]

def sortfn(items: list, key) -> list:
    return list(reversed(sorted(items, key=key)))