from flask import Flask, request, jsonify, render_template
from src import Gomoku, \
    Player, RandomPlayer, UCTPlayer, UCTQPlayer, \
    Model, ADPModel, \
    uct_pb_score, uct_score

app = Flask(__name__, static_url_path='/static')
game = None
player: Player = None

policies = {
    'uct_score': uct_score,
    'uct_pb_score': uct_pb_score,
}

models = {
    'ADPModel': ADPModel(),
}

players = {
    'RandomPlayer': RandomPlayer(),
    'UCTPlayer': UCTPlayer(policy=policies["uct_score"]),
    'UCTPBPlayer': UCTPlayer(policy=policies["uct_pb_score"]),
    'UCTQPlayer': UCTQPlayer(policy=policies['uct_score'], model=models["ADPModel"]),
    'UCTQPBPlayer': UCTQPlayer(policy=policies['uct_pb_score'], model=models["ADPModel"]),
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_game():
    data = request.json
    global game, player
    game = Gomoku(M=data.get("M"), N=data.get("N"), K=data.get("K"), FIRST_PLAYER=1)
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