let currentPlayer = 'X';
let rows = 3; // default
let cols = 3; // default
let winLength = 3; // default

function startGame() {
    // Get values from the form
    rows = parseInt(document.getElementById('rows').value);
    cols = parseInt(document.getElementById('cols').value);
    winLength = parseInt(document.getElementById('winLength').value);

    // Reset the board
    document.getElementById('board').innerHTML = '';
    currentPlayer = 'X';

    // Create the new board
    createBoard();
}

function createBoard() {
    const boardElement = document.getElementById('board');
    boardElement.style.gridTemplateColumns = `repeat(${cols}, 100px)`;

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const cellElement = document.createElement('div');
            cellElement.classList.add('cell');
            cellElement.id = `${i}${j}`;
            cellElement.onclick = () => cellClicked(`${i}${j}`);
            boardElement.appendChild(cellElement);
        }
    }
}

// function createBoard() {
//     const boardElement = document.getElementById('board');
//     boardElement.style.gridTemplateColumns = `repeat(${cols}, 100px)`;

//     for (let i = 0; i < rows; i++) {
//         const rowElement = document.createElement('div');
//         rowElement.classList.add('row');
//         for (let j = 0; j < cols; j++) {
//             const cellElement = document.createElement('div');
//             cellElement.classList.add('cell');
//             cellElement.id = `${i}${j}`;
//             cellElement.onclick = () => cellClicked(`${i}${j}`);
//             rowElement.appendChild(cellElement);
//         }
//         boardElement.appendChild(rowElement);
//     }
// }

function cellClicked(cellId) {
    const cell = document.getElementById(cellId);

    if (cell.innerHTML === '' && !checkWinner()) {
        cell.innerHTML = currentPlayer;
        if (checkWinner()) {
            alert(`Player ${currentPlayer} wins!`);
            startGame(); // Restart the game
        } else {
            currentPlayer = currentPlayer === 'X' ? 'O' : 'X';
        }
    }
}

function checkWinner() {
    const directions = [
        [0, 1],
        [1, 0],
        [1, 1],
        [1, -1],
    ];

    for (const [dx, dy] of directions) {
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                if (
                    i + (winLength - 1) * dx < rows &&
                    j + (winLength - 1) * dy < cols &&
                    i + (winLength - 1) * dx >= 0 &&
                    j + (winLength - 1) * dy >= 0
                ) {
                    let win = true;
                    for (let k = 0; k < winLength; k++) {
                        const ni = i + k * dx;
                        const nj = j + k * dy;
                        const cell = document.getElementById(`${ni}${nj}`);
                        if (cell.innerHTML !== currentPlayer) {
                            win = false;
                            break;
                        }
                    }
                    if (win) {
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

// Initial board creation
createBoard();
