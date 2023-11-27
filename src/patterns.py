lti = lambda c: 1 if c == 'x' else -1 if c == 'o' else 0
itl = lambda c: 'x' if c == 1 else 'o' if c == -1 else '-'
revp = lambda p: ''.join([itl(-lti(c)) for c in p])

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