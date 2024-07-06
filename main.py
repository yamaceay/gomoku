from typing import Callable
import random
import copy
import os
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.optim as optim

def softmax(x: np.ndarray) -> np.ndarray:
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def dirichlet_noise(x: int):
    return np.random.dirichlet([.03] * x)


def entropy_fn(log_act_probs: torch.Tensor) -> torch.Tensor:
    return -torch.mean(
        torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
    )
    
def policy_loss_fn(mcts_probs: torch.Tensor, log_act_probs: torch.Tensor) -> torch.Tensor:
    return -torch.mean(
        torch.sum(mcts_probs*log_act_probs, 1)
    )
    
def kl_divergence(old_probs: np.ndarray, new_probs: np.ndarray) -> float:
    return np.mean(
        np.sum(old_probs * (
            np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)
        ), axis=1)
    )
    
def explained_var(labels: np.ndarray, preds: np.ndarray) -> float:
    return 1 - np.var(labels - preds) / np.var(labels)

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

class CNN(nn.Module):
    def __init__(self, M: int, N: int):
        super().__init__()

        self.M = M
        self.N = N

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.policy_layers = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*self.M*self.N, self.M*self.N),
            nn.LogSoftmax(dim=1)
        )

        self.value_layers = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2*self.M*self.N, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, state_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_layers(state_input)
        x_policy = self.policy_layers(x)
        x_value = self.value_layers(x)
        return x_policy, x_value

class Zero_Net():
    def __init__(self, 
                 game_kwargs: tuple[int, int, int],
                 model_file: str = None, 
                 device: torch.DeviceObjType = torch.device('cpu'),
                 opt_args: dict = {}):
        self.device = device
        self.M, self.N, _ = game_kwargs
        self.opt_args = opt_args

        self.cnn = CNN(self.M, self.N).to(device)
        self.optimizer = optim.Adam(self.cnn.parameters(), **opt_args)

        if model_file:
            self.load_model(model_file)

    def forward(self, state: Gomoku) -> tuple[list[tuple[float, int]], float]:
        actions = sorted(state.actions())
        legal_positions = [a[0] * state.N + a[1] for a in actions]
        curr_state = state.encode().reshape(
                -1, 4, self.M, self.N)
        curr_state = self.torch_batch(curr_state)
        log_act_probs, value = self.cnn(curr_state)
        log_act_probs, value = log_act_probs.detach(), value.detach()
        act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value[0][0]
    
    def predict(self, state: Gomoku) -> tuple[list[tuple[float, tuple[int, int]]], float]:
        with torch.no_grad():
            act_probs, value = self.forward(state)
        act_probs = sortfn([(p, (a // state.N, a % state.N)) for a, p in act_probs])
        return act_probs, value
    
    def torch_batch(self, batch: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(np.ascontiguousarray(batch)).to(self.device)

    def load_model(self, model_file: str):
        net_params = torch.load(model_file)
        self.cnn.load_state_dict(net_params)

    def save_model(self, model_file: str):
        net_params = self.cnn.state_dict()
        torch.save(net_params, model_file)
        
class Player:
    def copy(self):
        return copy.deepcopy(self)
    
    def next_move_probs(self, game: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        raise NotImplementedError
    
    def next_move(self, 
                  state: Gomoku, 
                  epsilon: float = .0, 
                  get_probs: bool = False) -> tuple[int, int] | list[tuple[float, tuple[int, int]]]:
        
        probs_actions = self.next_move_probs(state)
        probs, actions = zip(*probs_actions)
        if epsilon != .0:
            probs += epsilon * (self.noise(len(probs)) - probs)

        probs /= sum(probs)
        
        action_i = np.random.choice(list(range(len(actions))), p=probs)
        action = actions[action_i]
        if get_probs:
            return action, zip(probs, actions)
        return action

class Rand_Player(Player):
    def next_move_probs(self, game: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        actions = game.actions()
        probs = np.random.random(len(actions))
        probs /= probs.sum()
        probs_actions = [(probs[i], actions[i]) for i in range(len(actions))]
        probs_actions = sortfn(probs_actions)
        return probs_actions

def uniform_probs(state: Gomoku):
    actions = state.actions()
    probs = np.ones(len(actions))/len(actions)
    return zip(probs, actions)

class Node(object):
    def __init__(self, parent = None, p: float = 1.0):
        self.parent = parent
        self.children = {}
        self.n = 0
        self.Q = 0
        self.p = p

    def expand(self, probs_actions: list[tuple[float, tuple[int, int]]]):
        for prob, action in probs_actions:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def select(self, C: float = 5):
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].ucb_score(C))

    def ucb_score(self, C: float = 5):
        exploitation_term = self.Q
        exploration_term = self.p * np.sqrt(self.parent.n) / (1 + self.n)
        return exploitation_term + C * exploration_term

    def is_terminal(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None
    
    def __repr__(self) -> str:
        return f"Node(Parent={self.parent}, # Children={len(self.children)}), N={self.n}, Q={self.Q}, P={self.p}"

class Tree(object):
    def __init__(self, 
                 iterations: int, 
                 policy_value_fn: Callable = None,
                 k_ucb: float = 5,
                 gamma: float = 1.0,
                 ):
        
        self.root = Node()
        self.k_ucb = k_ucb
        self.iterations = iterations
        self.gamma = gamma
        
        if policy_value_fn is None:
            self.policy_value_fn = lambda s: (uniform_probs(s), self.rollout(s))
        else:
            self.policy_value_fn = policy_value_fn

    def iterate(self, state: Gomoku):
        node = self.root
        while not node.is_terminal():
            action, node = node.select(self.k_ucb)
            state.play(action)
        
        player = state.player
        
        if state.fin():
            reward = state.score() * player
        
        else:
            probs_actions, reward = self.policy_value_fn(state)
            node.expand(probs_actions)
            reward *= player
        
        self.backpropagate(node, -reward)
    
    def rollout(self, state: Gomoku) -> float:
        while not state.fin():
            action = state.actions()[0]
            state.play(action)
        return state.score()
    
    def backpropagate(self, node: Node, reward: float):
        while node is not None:
            node.n += 1
            node.Q += self.gamma * (reward - node.Q) / node.n
            reward = -reward
            node = node.parent

    def get_move_probs(self, 
                       state: Gomoku, 
                       temp: float, 
                       ) -> list[tuple[float, tuple[int, int]]]:
        for _ in range(self.iterations):
            self.iterate(state.copy())
        
        actions, probs = zip(*[(act, node.n) for act, node in self.root.children.items()])
        probs = np.array(probs) 
        
        if self.policy_value_fn is not None:
            probs = np.log(probs + 1e-10)
            
        probs = softmax(probs / temp)
        
        return zip(probs, actions)

class Deep_Player(Player):
    def __init__(self, 
                 iterations: int, 
                 policy_value_fn: Callable = None, 
                 k_ucb: float = 5, 
                 temp: float = .001,
                 memory: bool = False,
                 ):
        
        self.noise = dirichlet_noise
        self.tree = Tree(
            iterations=iterations,
            policy_value_fn=policy_value_fn,
            k_ucb=k_ucb,
        )
        
        self.temp = temp
        self.history = []
        self.memory = memory

    def reuse_tree(self, state: Gomoku) -> bool:
        prev_history = list(self.history)
        self.history = list(state.history)
        if self.memory and len(self.history) > len(prev_history):
            invalid_match = False
            for h1, h2 in zip(prev_history, self.history):
                if h1 != h2:
                    invalid_match = True
                    break
            if not invalid_match:
                rest_history = self.history[len(prev_history):]
                not_found = False
                for move in rest_history:
                    if move not in self.tree.root.children:
                        not_found = True
                        break
                    self.tree.root = self.tree.root.children[move]
                    self.tree.root.parent = None
                if not not_found:
                    return True
        self.tree.root = Node()
        return False

    def next_move_probs(self, state: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        self.reuse_tree(state)
        move_probs = self.tree.get_move_probs(state, temp=self.temp)
        move_probs = sortfn(move_probs)
        return move_probs
    

class Zero_Player(Deep_Player):
    def __init__(self, *args, **kwargs):
        kwargs["memory"] = True
        assert kwargs["policy_value_fn"] is not None, "policy_value_fn has to be assigned"
        super().__init__(*args, **kwargs)

class Flat_Player(Player):
    def __init__(self, 
                 policy_value_fn: Callable,
                 temp: float = .001):
        
        self.policy_value_fn = policy_value_fn
        self.temp = temp
        
    def next_move_probs(self, state: Gomoku) -> list[tuple[float, tuple[int, int]]]:
        probs_actions, _ = self.policy_value_fn(state) 
        probs, actions = zip(*probs_actions)
        probs = np.array(probs) 
            
        probs = np.log(probs + 1e-10)
        probs = softmax(probs / self.temp)
        
        return zip(probs, actions)        

TRAIN_ARGS = {
    "6_6_4": dict(n_zero = 400, n_uct = 5000, n_uct_step = 1000, n_uct_max = 5000),
    "8_8_5": dict(n_zero = 500, n_uct = 4500, n_uct_step = 1500, n_uct_max = 6000),
    "10_10_5": dict(n_zero = 600, n_uct = 6000, n_uct_step = 2000, n_uct_max = 6000),
}

def find_best_level(game_kwargs_str: str) -> int:
    level = 0
    while os.path.exists(f"{game_kwargs_str}/models/v{level + 1}.pkl"):
        level += 1
    return level

def get_player(game_kwargs: tuple[int, int, int], name: str, level: int = 0) -> tuple[tuple[str, Player, float], bool]:
    game_kwargs_str = "_".join(map(str, game_kwargs))
    train_kwargs = TRAIN_ARGS["_".join(map(str, game_kwargs))]
    n_zero = train_kwargs["n_zero"]
    n_uct_step = train_kwargs["n_uct_step"]
    
    det = True
    if name == "UCT":
        assert level != -1, "UCT level must be specified"
        n_it = n_uct_step * level
        player = (f"{name}_{n_it}", Deep_Player(iterations=n_it), .0)    
        det = False
    else:
        if level == 0:
            level = find_best_level(game_kwargs_str)
        net = Zero_Net(
            game_kwargs=game_kwargs, 
            model_file=f"{game_kwargs_str}/models/v{level}.pkl",
        )
        if name == "FLAT":
            player = (f"{name}_v{level}", Flat_Player(policy_value_fn=net.predict), .0)
        elif name == "ZERO":
            player = (f"{name}_v{level}", Deep_Player(iterations=n_zero, policy_value_fn=net.predict), .0)
        elif name == "ZEROX": 
            player = (f"{name}_v{level}", Deep_Player(iterations=n_zero, policy_value_fn=net.predict, memory=True), .0)
    return player, det

class EndError(Exception):
    def __str__(self):
        return "Game ended"

class InfoError(Exception):
    def __init__(self, message):
        self.message = message
    
    def __str__(self):
        return self.message

class Bot:
    def __init__(self, *player_args, epsilon: float = .0):
        self.player_fn = lambda x: get_player(x, *player_args)[0][1]
        self.epsilon = epsilon

    def get_attr(self, attr):
        if hasattr(self, attr):
            return self.__dict__[attr]
        else:
            raise InfoError(f"Attribute {attr} not found")

    def run(self):
        while True:
            try:    
                message = input().strip()
                header, message = message.split()[0], " ".join(message.split()[1:])
                header = str.upper(header)
                if header == "INFO":
                    self.process_info(message)
                elif header == "START":
                    self.start_game()
                elif header == "TURN":
                    self.make_move(message)
                elif header == "END":
                    self.end_game()
            except Exception as e:
                print(f"ERROR {e}")
                continue

    def process_info(self, message):
        [key, value] = message.split()
        if key == "initial_board":
            self.load_initial_board(value)
            self.player = self.player_fn(self.game_kwargs)
        elif key == "match_name":
            self.log_file = f"{value}.log"
        elif key == "space_limit":
            pass
        elif key == "timeout_turn":
            pass
        else:
            raise InfoError(f"Key {key} not found")
        print("OK")

    def load_initial_board(self, filename):
        if not os.path.exists(filename):
            raise InfoError(f"File {filename} not found")
        with open(filename, 'r') as f:
            game_str = f.readlines()
        M = len(game_str)
        N = len(game_str[0].replace("\n", "").strip())
        K = 5 if M >= 7 and N >= 7 else 4
        self.game_kwargs = (M, N, K)
        game = Gomoku(*self.game_kwargs)
        moves_X, moves_O = [], []
        for i, line in enumerate(game_str):
            for j, c in enumerate(line.strip()):
                if c == 'X':
                    moves_X.append((i, j))
                elif c == 'O':
                    moves_O.append((i, j))
        for move_X, move_O in zip(moves_X, moves_O):
            game.play(move_X)
            game.play(move_O)
        if len(moves_X) > len(moves_O):
            game.play(moves_X[-1])
        self.initial_game = game
        self.game = self.initial_game.copy()

    def start_game(self):
        assert str(self.game) == str(self.initial_game), "Game state not equal to initial state"
        move = self.do_move()
        print(move)

    def save_move(self, message):
        if self.game.fin():
            raise EndError
        [opponent_move] = message.split()
        (x, y) = map(int, opponent_move.split('-'))
        self.game.play((x, y))

    def make_move(self, message):
        self.save_move(message)
        move = self.do_move()
        print(move)

    def do_move(self):
        if self.game.fin():
            raise EndError
        (x, y) = self.get_attr("player").next_move(self.game, self.epsilon)
        self.game.play((x, y))
        return f"{x}-{y}"

    def end_game(self):
        self.game = self.get_attr("initial_game").copy()
        print("OK")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--player", choices=["FLAT", "ZERO", "ZEROX"], required=True, help="Player type. FLAT: AlphaZero w/o MCTS, ZERO: AlphaZero, ZEROX: AlphaZero w/ memory.")
    parser.add_argument("--epsilon", type=float, default=.0, help="Noise parameter for any player, ranging from 0 to 1. Zero for deterministic players, one for random players. Defaults to 0.")
    parser.add_argument("--level", type=int, default=0, help="Strength of player, represented by 1, ..., MAX_LEVEL (= 0). Defaults to 0.")
    args = parser.parse_args()
    bot = Bot(args.player, args.level, epsilon=args.epsilon)
    bot.run()