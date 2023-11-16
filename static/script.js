let board;
let currentPlayer;
let M, N, K;
let deactivate = false;
let waiting = false;

async function startGame() {
    // Lesen Sie die Konfigurationswerte M, N und K
    M = parseInt(document.getElementById('rows').value);
    N = parseInt(document.getElementById('cols').value);
    K = parseInt(document.getElementById('winLength').value);
    player = document.getElementById('player').value;

    // Überprüfen Sie, ob die Werte gültig sind
    if (isNaN(M) || isNaN(N) || isNaN(K) || M <= 0 || N <= 0 || K <= 0) {
        alert('Bitte geben Sie gültige Werte für M, N und K ein.');
        return;
    }

    // Senden Sie eine POST-Anfrage an den Server, um das Spiel zu starten
    const response = await fetch('/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ M, N, K, player }),
    });
    const data = await response.json();
    if (data.success) {
        // Erstellen Sie ein neues Brett und starten Sie das Spiel
        board = Array(M).fill().map(() => Array(N).fill(''));
        currentPlayer = 'X';
        deactivate = false;
        const messageElement = document.getElementById('message');
        messageElement.textContent = '';
        drawBoard();
    } else {
        alert('Es gab einen Fehler beim Starten des Spiels.');
    }
}

function drawBoard() {
    const boardDiv = document.getElementById('board');
    boardDiv.innerHTML = '';
    boardDiv.style.gridTemplateColumns = `repeat(${N}, 1fr)`;
    boardDiv.style.gridTemplateRows = `repeat(${M}, 1fr)`;
    for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
            const cell = document.createElement('div');
            const move = [i, j];
            cell.textContent = board[i][j];
            cell.className = 'cell';
            cell.addEventListener('click', () => {
                if (waiting) return; // Überprüfen Sie, ob auf einen Zug gewartet wird
                const result = cellClicked(move);
                drawBoard();
                if (result) makeMove(move);
            });
            boardDiv.appendChild(cell);
        }
    }
}

async function makeMove(move) {
    waiting = true; // Setzen Sie waiting auf true, wenn ein Zug gemacht wird
    const response = await fetch('/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ move }),
    });
    const data = await response.json();
    waiting = false; // Setzen Sie waiting auf false, wenn eine Antwort vom Server empfangen wird
    if (data.move) {
        cellClicked(data.move);
        drawBoard();
    }

    if (data.game_over) {
        deactivate = true;
        let message;
        switch (data.winner) {
            case 1:
                message = 'Player X wins!';
                break;
            case 0:
                message = 'Tie break!';
                break;
            case -1:
                message = 'Player O wins!';
                break;
        }
        const messageElement = document.getElementById('message');
        messageElement.textContent = message;
    }
}

function cellClicked([i, j]) {
    if (deactivate || waiting) return; // Überprüfen Sie, ob auf einen Zug gewartet wird
    if (board[i][j] === '') {
        board[i][j] = currentPlayer;
        currentPlayer = currentPlayer === 'X' ? 'O' : 'X';
        return true;
    }
    return false;
}

document.addEventListener('DOMContentLoaded', (event) => {
    startGame();
});