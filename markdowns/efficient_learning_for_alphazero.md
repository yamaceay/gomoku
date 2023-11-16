In recent years, deep reinforcement learning have made great breakthroughs on board games. Still, most of the works require huge computational resources for a large scale of environmental interactions or self-play for the games. This paper aims at building powerful models under a limited amount of self-plays which can be utilized by a human throughout the lifetime. We propose a learning algorithm built on AlphaZero, wiht its path searching regularised by a path consistency optimality, i.e. values on one optimal search path should be identical. Thus, the algorithm is shortly named PCZero. In implementation, historical trajectory and scouted search paths by MCTS makes a good balance between exploration and exploitation, which enchances the generalization ability effectively. PCZero obtains 94.1% winning rate against the champion of Hex Computer Olympiad in 2015 on 13 x 13 hex, much higher than 84.3% by AlphaZero. The models consume only 900K self-play games, about the amount humans can study in a lifetime. The improvements by PCZero have been also generalized to Othello and Gomoku. Experiments also demonstrate the efficiency of PCZero under offline learning setting.



Tree search: Slow reasoning process
Deep reinforcement learning + heuristic search: Faster, but requires a large amount of self-play games
PCZero: 
* Each game is a playing path ( = sequential game states)
* Task: Predict the policy and the value of current state accurately
* Path consistency, a path optimality constraint, saying "values on one optimal path should be identical"
* Regularize path searching by adding a penalty term: $L(\theta) = L_{RL}(\theta) + \lambda L_{PC}(\theta)$
* It can be a L1, L2 deviation from an estimated value of optimal path which could be simply a moving average within a window of estimated optimal path

Proposal:
1. A data-efficient learning algorithm for games called PCZero, built on top of AlphaZero. The training process of AlphaZero is augmented by minimizing the violation of the PC condition. 
2. An effective implementation of PCZero by including the MCTS simulated paths into the computation for PC, in addition to the historical trajectory. The heuristics drawn from the lookahead search of MCTS is beneficial to the network learning in jumping out of the local optimum.
3. Extension of PC from the consistency of state values to the consistency of feature maps, which are taken from the network layers before the state value's final estimation. Feature consistency provides a stricter constraint on PC nature, and it works as a complement to the value consistency. 

### Related Work
Sequential games with perfect information: All possible game states can be included in a game tree starting at the initial state and containing all possible moves

A* search algorithm computes f(s) as the summation of g(s) accumulated cost/reward from s_0 to the preferred termination n
Path optimality: f(s) = f(n) for every node s on an optimal path and f(s) > f(n) for every node s not on an optimal path.

Rely on this optimality to use A* to make a lookahead scouting to estimate a segment on the optimal path and use the average of f-values from the root to n.
1. Estimate f(n) by deep neural networks
2. Make a lookahead scouting by MCTS instead of A*
3. Consider a moving average within a window of estimated optimal path in place of the entire of historical trajectory plus lookahead scouting

Q-Iteration -> Might be too heavy computation
Instead Q-Learning -> Bellmann Equation tells us if the maximum doesn't improve significantly any more, it should have learnt sufficiently. It converges to the optimal Q-function over time. 

Online learning: Interaction is expensive or infeasible, and the utilization of collected data is insufficient and requires a huge amount of experience to train a working model

Offline RL applied the supervised learning method is more sample-efficient, it utilizes only previously collected offline data without interactions. Important challenge: Answering counterfactual queries, generalizing model to the unexperienced situations.

### PC

Boards are typical delayed reward applications. Immediate reward r(s, a) is always zero and g(s) = 0 holds until arriving the termination state.
f(s) = h(s) -> equivalent to the state value v(s) in RL. f(s) is turned into the consistency of v(s). If optimal playing policy and state transition function are deterministic ( = $\pi(s)^* = \arg\max_{a \in A} r(s, a) + v(s'), s' = T(s, a)$ where T is a state transition function)
* $v^*(s) = r(s, \pi^*(s)) + \gamma v^*(s') = v^*(s')$

Path consistency is a condition that optimal state value $v^*$ should satisfy in RL. Objective function is penalized by path consistency, then the optimal state value $v^*$ is guaranteed to be consistent with the optimal path.

AlphaZero computes v(s) using a deep NN. PCZero is trained by further restricting that the estimated v(s) should deviate as less as possible along the optimal path
Realizing PC constraint:
1. Compute the variance of the values within a sliding window W, containing state s: $L_{PC}(s) = (v - \overline{v})^2$ where $\overline{v}$ is an average state value in $W_s$. The initial averages are insufficient and unreliable => first stages of training it ends up in a local optimum.

2. Calculate $\overline{v}$ using not only historical trajectory but also scouted heuristic path provided by MCTS while doing self-play. The farther the states from the roots, the more inconsistencies, it increases the variance of the value predictions within the window.

3. Relative (m upstream, n downstream length) of the heuristic path to the history path controls the balance between the satisfaction of optimal PC nature -> accelerates convergence, and randomness injection, avoids local optimum (see exploration vs. exploitation in RL). 

```python
def self_play(f, p):
    S = []
    s = initial_state
    while not s.is_terminal():
        if len(S) > l:
            S.pop(0)
        S.append(s)
        H, a = heuristic_path(s, f, p)
        v_bar = sum([f(s) for s in S + H]) / (len(S) + len(H))
        s = s.play(a)

def heuristic_path(s, f, p):
    H = []
    while not s.is_terminal() and len(H) <= k:
        a = argmax_a p(s, a)
        s = s.play(a)
        H.append(s)
    return H, a
```     

### Feature consistency

Feature maps of the value networks
-> Policy and value networks share a common residual tower to extract information from the board state. Value head performs convolution operation with one filter, yielding a feature map f before the final output layer. 

v is the linear combination of entries of f_v folowed by an activation function $\sigma$
* $v(s) = \sigma(Tr(w^T f_v))$

w is the weight matrix of the same size as the feature map f_v, Tr is the trace operator.

$L_{PC}^f = ||f_v - \overline{f_v}||^2$ where $\overline{f_v}$ is the average feature map over the nodes within the window. Feature consistency is sufficient but not necessary condition for value consistency. $L_{PC}^f$ is a tighter constraint

### Loss functions

$L_1 = -y^T \log p + (z - v)^2 + \lambda L_{PC} + \beta L_{PC}^f + c||\theta||^2$

$L_2 = -\pi^T \log p + (z - v)^2 + \lambda L_{PC} + \beta L_{PC}^f + c||\theta||^2$

where:
* p is the policy vector
* v is the value vector
* y is the one-hot action vector
* z denotes the final result of the game -1, 1
* $\lambda, \beta$ are the non-negative coefficients to adjust the influence of PC. 
* $||\theta||^2$ is the L2 regularization term

Policy-value network: 
* A convolutional head extracting input information
* n = 3 residual blocks with ReLU activation and batch normalization
* The extracted information is passed to a policy head and a value head respectively
* Input data: five binary planes
    * Position of the pieces: black stones (1), white stones (2), empty points (3)
    * The current game player: Black plays (4), white plays (5)

PUCT criterion with $c = 1.5$:
* $a = \arg\max_{a} \{Q(s, a) + c_{puct}p(s, a) \frac{\sqrt{N(s)}}{1 + N(s, a)}\}$

Dirichlet noise is added into nodes prior probability as a source of randomness. 