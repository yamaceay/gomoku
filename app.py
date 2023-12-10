from flask import Flask, request, jsonify, render_template
from src import Gomoku, \
    Player, RandomPlayer, \
    uct_score, UCT_Player, \
    ADP_Dense_Player, \
    AlphaZeroPlayer

app = Flask(__name__, static_url_path='/static')
game = None

game_kwargs = {
    'M': 8,
    'N': 8,
    'K': 5,
    'ADJ': 2,
}

adp_kwargs = {
    'alpha': 0.9,
    'gamma': 0.9,
    'lr': 0.01,
}

player: Player = None

players = {
    '_RANDOM': RandomPlayer(),
    '_ADP_v1': ADP_Dense_Player(model_path="_dens2/models/epoch_250.h5", game_kwargs=game_kwargs, **adp_kwargs),
    '_ADP_v2': ADP_Dense_Player(model_path="_dens2/models/epoch_500.h5", game_kwargs=game_kwargs, **adp_kwargs),
    '_ADP_v3': ADP_Dense_Player(model_path="_dens2/models/epoch_750.h5", game_kwargs=game_kwargs, **adp_kwargs),
    '_ADP_v4': ADP_Dense_Player(model_path="_dens2/models/epoch_1000.h5", game_kwargs=game_kwargs, **adp_kwargs),
    '_UCT_100': UCT_Player(iterations=100, policy=uct_score),
    '_UCT_1k': UCT_Player(iterations=1000, policy=uct_score),
    '_ALPHAZERO': AlphaZeroPlayer(game_kwargs=game_kwargs),
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
    move = tuple(move)
    score, game_over = game.play(move)
    move = None
    probs_actions = []
    
    if not game_over:
        probs_actions = player.next_move_probs(game)
        probs_actions = [(float(prob), list(action)) for prob, action in probs_actions]
        move = player.next_move(game)
        score, game_over = game.play(move)

    return jsonify(
        score=score, 
        game_over=game_over, 
        move=move,
        probs=probs_actions,
    )

if __name__ == '__main__':
    app.run(debug=True)