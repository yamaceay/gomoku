package main

import (
	"log"

	"github.com/gomoku/items"
)

func main() {
	board, err := items.NewBoard(4, 4, 3)
	if err != nil {
		log.Printf("couldn't create a board: %s", err)
		return
	}

	positions := []items.Position{
		{X: 0, Y: 0},
		{X: 1, Y: 0},
		{X: 0, Y: 1},
		{X: 1, Y: 1},
		{X: 0, Y: 2},
		{X: 1, Y: 2},
	}

	items.PlayGame(*board, positions)
}
