package items

import (
	"fmt"
	"strings"
)

type Player int

func (p Player) Stone() string {
	stone := "-"
	if p == Black {
		stone = "X"
	} else if p == White {
		stone = "O"
	}
	return stone
}

func (p Player) String() string {
	var str string
	if p == Black {
		str = "Black"
	} else if p == White {
		str = "White"
	}
	return str
}

type Position struct {
	X int
	Y int
}

const Black = 1
const White = -1

var allDirections = []Position{{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}

func PlayGame(board Board, positions []Position) (Board, string, error) {
	var output string
	for i, p := range positions {
		if err := (&board).Move(p); err != nil {
			return board, output, fmt.Errorf("Player %d unable to make the move: %v", board.Player, p)
		}
		output += sprintStatus(board, i, p)
		if ended, winner := (&board).End(); ended {
			output += fmt.Sprintf("Winner is: %s", winner)
			break
		}
	}
	return board, output, nil
}

func NewBoard(M int, N int, K int) (*Board, error) {
	var board Board
	if M < 0 {
		return &board, fmt.Errorf("m out of bounds (<0)")
	} else if N < 0 {
		return &board, fmt.Errorf("n out of bounds (<0)")
	} else if K < 0 {
		return &board, fmt.Errorf("k out of bounds (<0)")
	} else if M < K || N < K {
		return &board, fmt.Errorf("k too large (>M, >N)")
	}

	return &Board{
		Player: Black,
		Stones: make(map[Position]Player),
		M:      M,
		N:      N,
		K:      K,
	}, nil
}

func sprintStatus(board Board, i int, p Position) string {
	summary := fmt.Sprintf("%d: %s -> (%d, %d)", i+1, -board.Player, p.X, p.Y)
	return fmt.Sprintf("%s:\n\n%s\n\n", summary, board)
}

type Board struct {
	Player
	Stones map[Position]Player
	M      int
	N      int
	K      int
}

func (b Board) String() string {
	whole := []string{}
	for y := 0; y < b.N; y++ {
		row := []string{}
		for x := 0; x < b.M; x++ {
			position := Position{x, b.M - 1 - y}
			player := b.Stones[position]
			stone := player.Stone()
			row = append(row, stone)
		}
		rowString := strings.Join(row, " | ")
		whole = append(whole, rowString)
	}
	lines := strings.Repeat("-", 4*(b.N)-2)
	wholeString := strings.Join(whole, fmt.Sprintf("\n%s\n", lines))
	return wholeString
}

func (b *Board) Move(p Position) error {
	if p.X < 0 || p.X > b.M-1 {
		return fmt.Errorf("x out of bounds")
	} else if p.Y < 0 || p.Y > b.N-1 {
		return fmt.Errorf("y out of bounds")
	}

	if b.Stones[p] != 0 {
		return fmt.Errorf("given position is not empty")
	}

	b.Stones[p] = b.Player
	b.Player = -b.Player

	return nil
}

func (b *Board) Iterate(s Position, d Position) ([]Player, error) {
	players := []Player{}

	si := Position{s.X, s.Y}
	for i := 0; i < b.K; i++ {
		if si.X < 0 || si.X > b.M-1 {
			return players, fmt.Errorf("x out of bounds")
		}
		if si.Y < 0 || si.Y > b.N-1 {
			return players, fmt.Errorf("y out of bounds")
		}
		player := b.Stones[si]
		players = append(players, player)

		si = Position{si.X + d.X, si.Y + d.Y}
	}

	return players, nil
}

func (b *Board) End() (bool, Player) {
	for x := 0; x < b.M-1; x++ {
		for y := 0; y < b.N-1; y++ {
			position := Position{x, y}
			for _, direction := range allDirections {
				players, err := b.Iterate(position, direction)
				if err != nil {
					continue
				}
				if ended, winner := checkEnd(players); ended {
					return ended, winner
				}
			}
		}
	}
	return len(b.Stones) == b.M*b.N, 0
}

func checkEnd(players []Player) (bool, Player) {
	blackWins, whiteWins := true, true
	for _, player := range players {
		if player != Black {
			blackWins = false
		}
		if player != White {
			whiteWins = false
		}
	}
	if blackWins {
		return true, Black
	}
	if whiteWins {
		return true, White
	}
	return false, 0
}
