package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/gomoku/items"
)

func main() {
	board, err := items.NewBoard(4, 4, 3)
	if err != nil {
		log.Printf("failed to create a board: %s", err)
		return
	}

	positions, err := readPositions("play.txt")
	if err != nil {
		log.Printf("failed to read the positions: %s", err)
		return
	}

	_, output, _ := items.PlayGame(*board, positions)

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, output)
	})

	print("Listening on port 8080")
	http.ListenAndServe(":8080", nil)
}

func readPositions(filename string) ([]items.Position, error) {
	fp, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	content, err := io.ReadAll(fp)
	if err != nil {
		return nil, fmt.Errorf("failed to read content: %w", err)
	}
	var positions []items.Position
	for i, line := range strings.Split(string(content), "\n") {
		if len(line) == 0 {
			continue
		}
		coordinates := strings.Split(line, ",")
		if len(coordinates) != 2 {
			return nil, fmt.Errorf("failed to extract x and y from line %d", i)
		}
		x, _ := strconv.Atoi(coordinates[0])
		y, _ := strconv.Atoi(coordinates[1])
		position := items.Position{X: x, Y: y}
		positions = append(positions, position)
	}
	return positions, nil
}
