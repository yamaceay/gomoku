## History of Game Solvers

### Proof Number Search
PN Search is aimed exclusively at exhaustively searching the game tree. It is a best-first search algorithm that uses a heuristic evaluation function to guide the search. The heuristic evaluation function is used to estimate the number of moves required to win the game from a given position. The algorithm is based on the following two theorems:

* Theorem 1: If the proof number of a node is 0, then the node is a losing node.
* Theorem 2: If the disproof number of a node is 0, then the node is a winning node.

### Dependency-based search [NOT FOUND]

### Threat-space search
A threat is an important notion; the main types have descriptive names: the four, the straight four, the three, the broken three. To win the game against any opposition, a player needs to create a double threat (either a straight four, or two separate threats). By heavily relying on human expert analysis, threat-space search is able to reduce the search space and improve the search efficiency.

### Minimax tree search

Minimax algorithm was introduced by von Neumann and widely used in game theory and artificial intelligence. It is a decision rule used for minimizing the possible loss for a worst case scenario. It is used to evaluate the best move in two-player zero-sum games, and utilizes the whole search tree.

### Alpha-Beta Pruning

Alpha-beta pruning is a search algorithm that seeks to decrease the number of nodes that are evaluated by the minimax algorithm in its search tree. It is an adversarial search algorithm used commonly for machine playing of two-player games (Tic-tac-toe, Chess, Go, etc.). Alpha-beta pruning eliminates the need to evaluate subtrees of moves which are known to be worse than moves that have already been examined. But still in the worst case, it performs a full search on the game tree.

### Monte-Carlo Tree Search

Monte Carlo Tree Search is a search algorithm which expands a search tree dynamically using a evaluation heuristic as the selection policy. It consists of four steps:

1. Selection: Choose one of the best nodes worth exploring in the tree, for example: select from the 1-grid or 2-grid adjacent empty child nodes. The heuristic is typically chosen to be UCB: $v_i + k * \sqrt{\frac{\ln N}{n_i}}$ with $k = \sqrt{2}$, $v_i$ as the node value ( = total win rate of the child nodes), $N$ as the total number of simulations, $n_i$ as the total number of simulations of the node $i$.

2. Expansion: Create a new child node of the selected node

3. Simulation: Play out a game from the new expanded child node until arriving at a certain outcome. If the branching factor and search depth is too large, the complexity increases exponentially -> it is infeasible to simulate, so replace it with an evaluation function trained by ADP.

4. Backpropagation: Backpropagate the score of the node from teh previous expansio to all the previous parent nodes, and update the winning values and visit times of these nodes to facilitate the calculation of the UCB value.

### Adaptive Dynamic Programming (ADP)

Adaptive Dynamic Programming is a method of reinforcement learning which is able to learn from the environment without knowing the Markov decision process. It is a method of solving complex problems by approximating the cost function $V$ and the policy function $p$ in the Bellmann equation. Using a three-layer fully connected NN, the final output is:

* $h_i(t) = \sum_{j = 1}^n x_j(t) w_{ji}^{(1)}(t)$
* $g_i(t) = \frac{1}{1 + \exp(-h_i(t))}$
* $p(t) = \sum_{i = 1}^{m} w_i{(2)} (t)g_i(t)$

### UCT + ADP Algorithm

ADP is used to generate candidate moves for MCTS, which should be the root node corresponding to each progress of MCTS. Not only does it ensure the accuracy of the search, but also reduces the width of search. Compared with only using MCTS, it should save a large amount of time to find out the suitable action for Gomoku.

### UCT + ADP Progressive-Bias Algorithm

The algorithm learns the cost function $V$ by ADP, and uses the cost function to evaluate the value of the node in the MCTS tree. The algorithm selects the best move by the weighted sum of the winning rates calculated by MCTS and ADP as follows:

* UCB with incorporated heuristic: $UCB = v_i + k_1 * \sqrt{\frac{\ln N}{n_i}} + k_2 * \frac{H_i}{\max H}$


### AlphaZero

AlphaZero is a new method for learning to play the game of Go, chess and shogi. Starting from random play, and given no domain knowledge except the game rules, AlphaZero achieved within 24 hours a superhuman level of play in the games of chess and shogi as well as Go, and convincingly defeated a world-champion program in each case. AlphaZero is a more generic version of the AlphaGo Zero algorithm. It replaces the handcrafted knowledge and domain-specific augmentations used in traditional game-playing programs with deep neural networks, a general-purpose reinforcement learning algorithm, and a general-purpose tree search algorithm. Each of the neural networks is trained from scratch. The input to the neural networks is the board representation for the game of interest, and the output is a vector of move probabilities and a scalar value representing the expected outcome of the game.

AlphaZero - in general - employs two NNs:
1. Policy Network to make decisions during gameplay: It takes the current game state as input and outputs a probability distribution over all possible legal moves.
2. Value Network to evaluate board positions: It takes the current game state as input and predicts the probability of winning from that position.

### AlphaZero With Path Consistency

The regular training process can be improved by a path optimality constraint called path consistency, which indicates whether values on one optimal search path should be identical. The loss function is modified to include the path consistency loss, so that the estimated v(s) deviates as less as possible along the optimal path. More specifically, path optimality tells that $f(s) = f(n)$ for every node s on an optimal path and f(s) > f(n) for every node s not on an optimal path. This algorithm proposes three developments:

1. First, f(n) is estimated by NNs.
2. Second, a lookahead scouting is made by MCTS instead of A^*.
3. Third, a moving average within a window of estimated optimal path is considered in place of the entire of historical trajectory plus lookahead scouting. 