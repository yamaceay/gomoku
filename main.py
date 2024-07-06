from bot import Bot
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--player", choices=["FLAT", "ZERO", "ZEROX"], required=True, help="Player type. FLAT: AlphaZero w/o MCTS, ZERO: AlphaZero, ZEROX: AlphaZero w/ memory.")
    parser.add_argument("--epsilon", type=float, default=.0, help="Noise parameter for any player, ranging from 0 to 1. Zero for deterministic players, one for random players. Defaults to 0.")
    parser.add_argument("--level", type=int, default=0, help="Strength of player, represented by 1, ..., MAX_LEVEL (= 0). Defaults to 0.")
    args = parser.parse_args()
    bot = Bot(args.player, args.level, epsilon=args.epsilon)
    bot.run()