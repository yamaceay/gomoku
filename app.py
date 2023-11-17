from flask import Flask, request, jsonify, render_template
from src import Gomoku, \
    Player, RandomPlayer, UCTPlayer, UCTQPlayer, \
    Model, ADPModel, \
    pb_score, uct_pb_score, uct_score

app = Flask(__name__, static_url_path='/static')
game = None
player: Player = None

policies = {
    'uct_score': uct_score,
    'uct_pb_score': uct_pb_score,
    'pb_score': pb_score,
}

models = {
    'ADPModel': ADPModel(),
}

players = {
    '_': RandomPlayer(),
    '_UCT_UCB': UCTPlayer(policy=policies["uct_score"]),
    '_UCT_UCB_ADJ': UCTPlayer(policy=policies["uct_score"], tree_kwargs={"only_adjacents": True}),
    '_UCT_PB': UCTPlayer(policy=policies["pb_score"]),
    '_UCT_PB_ADJ': UCTPlayer(policy=policies["pb_score"], tree_kwargs={"only_adjacents": True}),
    '_UCT_UCB_PB': UCTPlayer(policy=policies["uct_pb_score"]),
    '_UCT_UCB_PB_ADJ': UCTPlayer(policy=policies["uct_pb_score"], tree_kwargs={"only_adjacents": True}),
    '_UCT_UCB_Q': UCTQPlayer(policy=policies['uct_score'], model=models["ADPModel"]),
    '_UCT_UCB_ADJ_Q': UCTQPlayer(policy=policies['uct_score'], tree_kwargs={"only_adjacents": True}, model=models["ADPModel"]),
    '_UCT_PB_Q': UCTQPlayer(policy=policies['pb_score'], model=models["ADPModel"]),
    '_UCT_PB_ADJ_Q': UCTQPlayer(policy=policies['pb_score'], tree_kwargs={"only_adjacents": True}, model=models["ADPModel"]),
    '_UCT_UCB_PB_Q': UCTQPlayer(policy=policies['uct_pb_score'], model=models["ADPModel"]),
    '_UCT_UCB_PB_ADJ_Q': UCTQPlayer(policy=policies['uct_pb_score'], tree_kwargs={"only_adjacents": True}, model=models["ADPModel"]),
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_game():
    data = request.json
    global game, player
    game = Gomoku(M=data.get("M"), N=data.get("N"), K=data.get("K"), FIRST_PLAYER=1, ADJ=data.get("ADJ"))
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
        
    return jsonify(score=score, game_over=game_over, move=move, winner=game.winner)

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(board=game.board.tolist(), player=game.player, winner=game.winner)

if __name__ == '__main__':
    app.run(debug=True)