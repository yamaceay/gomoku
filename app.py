from flask import Flask, request, jsonify, render_template
from src import Gomoku, \
    Player, RandomPlayer, ADP_Player, \
    AlphaZeroPlayer, \
    ValueNetwork, PolicyNetwork
    # UCT_Player, UCT_ADP_Player, \
    # pb_score, uct_pb_score, uct_score, \

app = Flask(__name__, static_url_path='/static')
game = None

game_kwargs = {
    'M': 8,
    'N': 8,
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

player: Player = None

# policies = {
#     'uct_score': uct_score,
#     'uct_pb_score': uct_pb_score,
#     'pb_score': pb_score,
# }

players = {
    '_RANDOM': RandomPlayer(),
    '_ADP': ADP_Player(value_network_kwargs, policy_network_kwargs),
    '_ALPHAZERO': AlphaZeroPlayer(**game_kwargs),
    # '_UCT_UCB': UCT_Player(timeout_ms=5000, policy=policies["uct_score"]),
    # '_UCT_UCB_ADJ': UCT_Player(timeout_ms=5000, policy=policies["uct_score"], tree_kwargs={"only_adjacents": True}),
    # '_UCT_PB': UCT_Player(timeout_ms=5000, policy=policies["pb_score"]),
    # '_UCT_PB_ADJ': UCT_Player(timeout_ms=5000, policy=policies["pb_score"], tree_kwargs={"only_adjacents": True}),
    # '_UCT_UCB_PB': UCT_Player(timeout_ms=5000, policy=policies["uct_pb_score"]),
    # '_UCT_UCB_PB_ADJ': UCT_Player(timeout_ms=5000, policy=policies["uct_pb_score"], tree_kwargs={"only_adjacents": True}),
    # '_UCT_ADP1_UCB': UCT_ADP_Player(timeout_ms=5000, max_depth=10, policy=policies['uct_score'], model=models["ADP1"]),
    # '_UCT_ADP1_UCB_ADJ': UCT_ADP_Player(timeout_ms=5000, max_depth=10, policy=policies['uct_score'], tree_kwargs={"only_adjacents": True}, model=models["ADP1"]),
    # '_UCT_ADP1_PB': UCT_ADP_Player(timeout_ms=5000, max_depth=10, policy=policies['pb_score'], model=models["ADP1"]),
    # '_UCT_ADP1_PB_ADJ': UCT_ADP_Player(timeout_ms=5000, max_depth=10, policy=policies['pb_score'], tree_kwargs={"only_adjacents": True}, model=models["ADP1"]),
    # '_UCT_ADP1_UCB_PB': UCT_ADP_Player(timeout_ms=5000, max_depth=10, policy=policies['uct_pb_score'], model=models["ADP1"]),
    # '_UCT_ADP1_UCB_PB_ADJ': UCT_ADP_Player(timeout_ms=5000, max_depth=10, policy=policies['uct_pb_score'], tree_kwargs={"only_adjacents": True}, model=models["ADP1"]),
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_game():
    data = request.json
    global game, player
    game = Gomoku(M=data.get("M"), N=data.get("N"), K=data.get("K"), ADJ=data.get("ADJ"))
    player = players[data.get("player")]
    return jsonify(success=True)

@app.route('/move', methods=['POST'])
def make_move():
    move = request.json.get('move')
    score, game_over = game.play(move)
    move = None
    
    if not game_over:
        move = player.next_move(game)
        score, game_over = game.play(move)
        
    return jsonify(
        score=score, 
        game_over=game_over, 
        move=move, 
        winner=game.winner
    )

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(
        board=game.board.tolist(), 
        player=game.player, 
        winner=game.winner
    )

if __name__ == '__main__':
    app.run(debug=True)