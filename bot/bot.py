from src import Gomoku, get_player
import os

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