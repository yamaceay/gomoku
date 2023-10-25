We combine Adaptive Dynamic Programming, a reinforcement learning method and UCB applied to trees algorithm witha  more powerful heuristic function based on Progressive Bias method and two pruning strategies for a traditional board game Gomoku. For the Adaptive Dynamic Programming part, we train a shallow forward neural network to give a quick evaluation of Gomoku board situations. UCT is a general approach in MCTS as a tree policy. Our framework use UCT to balance the exploration and exploitation of Gomoku game trees while we also apply powerful pruning strategies and heuristic function to re-select the available 2-adjacent grids of the state and use ADP instead of simulation to give estimated values of expanded nodes. Experiment result shows that this method can eliminate the search depth defect of the simulation process and converge to the correct value faster than single UCT. This approach can be applied to design new Gomoku AI and solve other Gomoku-like board game.

Techniques:
* UCT: Upper Confidence Bound applied to Monte-Carlo Tree Search
* ADP: Adaptive Dynamic Programming
* (More Advanced) Pruning Techniques:
    * VCF: Victory of Continuous Four
    * VCT: Victory of Continuous Three

### Introduction

* Minimax Tree Search: Whole extensive tree search
* Alpha-Beta Pruning: An improvement of Minimax, is able to prune the tree search by elimininating the branches that are not optimal for the current player. However, in worst case, it becomes Minimax.
* MCTS: As an enhancement of Alpha-Beta, it only expands the tree in the direction of the most promising nodes, balancing the exploration and exploitation. However, it still has a huge time and space complexity increasing exponentially with the simulation search depth
* ADP: An approach based on RL which doesn't need info about the Markov decision process. Take actions in an unknown env
* MCTS + ADP: Instead of simulating each node of any depth, it restricts the search depth to a certain level and uses a neural network to evaluate the value of the node. 

### Related Work

Browne applied MCTS methods to Gomoku.
Kang used Progressive Bias (a combination between UCB and Gomoku heuristic) -> increased the sampling accuracy and saved computational time
Wang and Mohandas applied Genetic Algorithms -> optimized tree search
UCT-RAVE, SO-ISMCTS, MO_ISMCTS
Zhao applied ADP with self-plays
Tang combined (winning probability in) MCTS and (weighted sum of) ADP.

### MCTS + UCB

MCTS can be a best-first search by precessing a large number of random simulation -> converges to the best solution

1. Selection: Choose one of the best nodes worth exploring in the tree, for example: select from the 1-grid or 2-grid adjacent empty child nodes, utilize Upper Confidential policy: $v_i + k * \sqrt{\frac{\ln N}{n_i}}$ with $k = \sqrt{2}$, $v_i$ as the node value ( = total win rate of the child nodes), $N$ as the total number of simulations, $n_i$ as the total number of simulations of the node $i$. 
2. Expansion: Create a new child node of the selected node
3. Simulation: Play out a game from the new expanded child node until arriving at a certain outcome. If the branching factor and search depth is too large, the complexity increases exponentially -> it is infeasible to simulate, so replace it with an evaluation function trained by ADP.
4. Backpropagation: Backpropagate the score of the node from teh previous expansio to all the previous parent nodes, and update the winning values and visit times of these nodes to facilitate the calculation of the UCB value.


### UCT-ADP Progressive-Bias Algorithm

Weighted sum of winning rates to select the next Gomoku sign position -> Speed and convergence remains unchanged, so use progressive bias to combine UCB with a priori exponential heuristic function and reconstruct the UCT tree structure, which is able to realize select, expand and simulate until a certain depth. Backpropagation: If no winner, return and update the final winning rate calculated by ADP.

1. Exponential Heuristic Function
$H_i = \sum \{(L_{open}^2 + (\frac{L_{close}}{2})^2)\}$ where:
* $L_{open}$: length of line that has no opponent's chessman at two terminals
* $L_{close}$: length of line that has only one opponent's chessman at its terminals
-> But discontinuous offense patterns are not considered, and valuation of offense is unreasonable. 

New heuristic function:
$H_i = \sum \{10^{L_{open}} * factor^j + 10^{L_{close} - 1} * factor^k\}$ where optimally $factor = 0.90$

$UCB = v_i + k_1 * \sqrt{\frac{\ln N}{n_i}} + k_2 * \frac{H_i}{\max H}$

with incorporated heuristic

2. Selecting Moves Pre-select Function
Use suitable adjacent grids moves to replace all legal moves -> 1 misses some significant points, therefore 2

3. Victory of Continuous Four and Victory of Continuous Three

VCF: Continuously manufacturing a sleep four attack in the case of the opponent do not have opportunity to fight back, until obtain the final winninng state or cannot generate new VCF situation. 

VCT: Continuously manufacturing a alive three attack in the case of the opponent do not have opportunity to fight back, until arriving at the final five or cannot generate new VCT or VCF circumstance

So: Human-designed heuristics function

4. Adaptive Dynamic Programming

Multi-Layer Perceptron of 3 layers is used. Input layer: the number of the 32 patterns and a one hot vector indicating who is to move next. # patterns are encoded in a specific way, number is represented by a vector of size 5. Activation: Sigmoid. Reward is set to 0 during the game, and the end result is either 1, 0, or -1. The prediction error is: $e(t) = \alpha[r(t + 1) + \gamma(t + 1) - V(t)]$, $\alpha$ is the learning rate and $\gamma$ is the discount factor set to 1. V is the output of the MLP. # of patterns is 32 -> 10k games for the MLP to converge. 

Uniform random moves -> Large branching factor and depth
Complex heuristic knowledge -> Computational cost per simulation increases

-> MLP trained by ADP to evaluate board situations instead of simulating play-outs to get winning or losing outcomes. 

### Convergence of UCT-ADP

