# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.M = int(kwargs.get('M', 8))
        self.N = int(kwargs.get('N', 8))
        self.K = int(kwargs.get('K', 5))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.M < self.K or self.N < self.K:
            raise Exception('board M and N can not be '
                            'less than {}'.format(self.K))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.M * self.N))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.M
        w = move % self.M
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.M + w
        if move not in range(self.M * self.N):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*M*N
        """

        square_state = np.zeros((4, self.M, self.N))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.M,
                            move_curr % self.N] = 1.0
            square_state[1][move_oppo // self.M,
                            move_oppo % self.N] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.M,
                            self.last_move % self.N] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        M = self.M
        N = self.N
        states = self.states
        n = self.K

        moved = list(set(range(M * N)) - set(self.availables))
        if len(moved) < self.K *2-1:
            return False, -1

        for m in moved:
            h = m // M
            w = m % M
            player = states[m]

            if (w in range(M - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(N - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * M, M))) == 1):
                return True, player

            if (w in range(M - n + 1) and h in range(N - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (M + 1), M + 1))) == 1):
                return True, player

            if (w in range(n - 1, M) and h in range(N - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (M - 1), M - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.state = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        M = board.M
        N = board.N

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(M):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(N - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(M):
                loc = i * M + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.state.init_board(start_player)
        p1, p2 = self.state.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.state, player1.player, player2.player)
        while True:
            current_player = self.state.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.next_move(self.state)
            self.state.move(move)
            if is_shown:
                self.graphic(self.state, player1.player, player2.player)
            end, winner = self.state.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.state.init_board()
        p1, p2 = self.state.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.next_move(self.state,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.state.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.state.current_player)
            # perform a move
            self.state.move(move)
            if is_shown:
                self.graphic(self.state, p1, p2)
            end, winner = self.state.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
