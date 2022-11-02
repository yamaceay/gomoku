package items

import (
	"fmt"
	"strings"
)

type Player int

type Position struct {
	X int
	Y int
}

const Black = 1
const White = -1

var allDirections = []Position{{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}

func PlayGame(board Board, positions []Position) error {
	for i, p := range positions {
		if err := board.Move(p); err != nil {
			return fmt.Errorf("Player %d unable to make the move: %v", board.Player, p)
		} else if ended, winner := board.End(); ended {
			fmt.Printf("Winner is: %d", winner)
			break
		}
		var stone string
		if board.Player == Black {
			stone = "X"
		} else if board.Player == White {
			stone = "O"
		}
		summary := fmt.Sprintf("[%d] %s plays (%d, %d): ", i+1, stone, p.X, p.Y)
		fmt.Printf("%s\n\n%s\n\n", summary, board)
	}
	return nil
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
			stone := "-"
			if player == Black {
				stone = "X"
			} else if player == White {
				stone = "O"
			}
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
	if p.X < 0 {
		return fmt.Errorf("x out of bounds (<0)")
	} else if p.Y < 0 {
		return fmt.Errorf("y out of bounds (<0)")
	} else if xBound := b.M - 1; p.X > xBound {
		return fmt.Errorf("x out of bounds (>%d)", xBound)
	} else if yBound := b.N - 1; p.Y > yBound {
		return fmt.Errorf("y out of bounds (>%d)", yBound)
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
