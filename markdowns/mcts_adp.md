Inspired by the core idea of AlphaGo, we combine a neural network, which is trained by Adaptive Dynamic Programming (ADP), with Monte Carlo Tree Search algorithm for Gomoku. MCTS algorithm is based on Monte Carlo simulation method, which goes through lots of simulations and generates a game search tree. We rollout it and search the outcomes of the leaf nodes in the tree. As a result, we obtain the MCTS winning rate. The ADP and MCTS methods are used to estimate the winning rates respectively. We weight the two winning rates to select the action position with the maximum one. Experiment result shows that this method can effectively eliminate the neural network evaluation function's "short-sighted" defect. With our proposed method, the game's final prediction result is more accurate, and it outperforms the Gomoku with ADP algorithm. 


### Introduction

Proof-based search
Dependency-based search
Thread-space search
Game tree searching -> Minimax tree
William: A complete search to a depth of n moves requires evaluations of p!/(n-p)! board situations (p: # legal moves, n: depth)
Alpha-beta search speeds up a bit

### Related Work

Freisleben: NN which has the capacity to learn to play Gomoku
Train a NN by rewarding / penalizing from a special RL algorithm (comparison training)
No need to know about the Markov decision process. 

Mo, Gong: Temporal difference learning as an enhancement. 
Continuous value function approximated by a nonlinear function line, core idea of ADP.

Pair it with a tree-layer fully connected NN to provide adaptive and self-teaching behavior.

MCTS: finds optimal decisions in a given domain by taking random simulations in the decisiion space and building a search tree according to the results. 

HMCTS:

f is a function to generate a new board state from last board state and action. Heuristic knowledge which is common knowledge for Gomoku players can save more time in simulation than random sampling. Therefore, it helps the result getting converge earlier than before. The rules are explained as follows:

1. If there is a continuous four, forced to move its piece to the position where it can emerge five-in-a-row in my side.
2. If four-in-a-row in opposite side, block it.
3. If three-in-row in my side, move my piece to the position where it can emerge four-in-a-row.
4. Three-in-a-row oppositely, block it as well. 

$Q(s, a) = \frac{1}{N(s, a)} \sum_{i = 1}^{N(s)} l_i(s, a) z_i$

where $N(s, a)$ is the number of times that action $a$ is taken in state $s$, $N(s)$ is the number of times that state $s$ is visited, $l_i(s, a)$ is whether the action $a$ is taken in the game $i$ in state $s$, $z_i$ is the result of the game $i$.

Alternatively, UCT:

$UCB = \overline{x_j} + \sqrt{\frac{2 \ln N}{n_j}}$

where $\overline{x_j}$ is the average reward of the j-th child node, $N$ is the total number of simulations, $n_j$ is the number of simulations of the j-th child node.

UCT is HMCTS applied using UCB as heuristic.  

... 

Adaptive Dynamic Programming With MCTS

Trained by temporal difference learning. Obtain candidate action moves by ADP, which should be the root node corresponding to each progress of MCTS = Not only does it ensure the accuracy of the search, but also reduces the width of search. Compared with only using MCTS, it should save a large amount of time to find out the suitable action for Gomoku. 

Current board state is fed as input -> (Action Selection) -> Control action is generated

Then, next step transition state is obtained -> (Utility function) -> Reward is obtained. 

Estimate the cost function V, so that the Bellmann equation is satisfied. Using a three-layer fully connected NN. Final output is:
* $h_i(t) = \sum_{j = 1}^n x_j(t) w_{ji}^{(1)}(t)$
* $g_i(t) = \frac{1}{1 + \exp(-h_i(t))}$
* $p(t) = \sum_{i = 1}^{m} w_i{(2)} (t)g_i(t)$
* $v(t) = \frac{1}{1 + \exp(-p(t))}$

where $x_j(t)$ is the input of the j-th neuron in the input layer, $w_{ji}^{(1)}(t)$ is the weight between the j-th neuron in the input layer and the i-th neuron in the hidden layer, $w_i{(2)} (t)$ is the weight between the i-th neuron in the hidden layer and the output neuron, $g_i(t)$ is the output of the i-th neuron in the hidden layer, $p(t)$ is the input of the output neuron, $v(t)$ is the output of the output neuron.

Training:

Firstly, obtain 5 candidate moves and their winning probabilities from the NN which is trained by ADP. 
Secondly, 5 candidate moves and their situations of board are seen as the root node of MCTS.
Obtain 5 winning probabilities respectively from MCTS method.
Thirdly, calculate the weighted sum of ADP and its corresponding winning probability of MCTS:

* $w_p = \lambda w_1 + (1 - \lambda) w_2$

```latex
Algorithm 3: ADP with MCTS
input original state s0;
output action a correspond to ADP with MCTS; MADP, WADP ← ADP Stage(s0);
WMCTS ← MCTS Stage(MADP);
for each w1, w2 in pairs(WADP, WMCTS) do
wp ← λw1 + (1-λ)w2;
add p into P;
end for each
return action a correspond to max p in P
ADP Stage(state s)
obtain top 5 winning probability WADP from ADP(s) ; obtain their moves MADP correspond to WADP;
return MADP, WADP
MCTS Stage(moves MADP)
for each move m in MADP do
create m as root node with correspond state s obtain w2 from MTCS(m, s)
add w2 into WMCTS
end for each return WMCTS
```

```python
def mcts_adp(state, c):
    madp, wadp = adp(state)
    wmcts = []
    for move in madp:
        wmcts.append(mcts(move, state))
    p = []
    for w1, w2 in zip(wadp, wmcts):
        p.append(c * w1 + (1 - c) * w2)
    return max(p)

def adp_stage(state):
    wadp, madp = adp(state)
    return wadp, madp

def mcts_stage(moves):
    wmcts = []
    for move in moves:
        wmcts.append(mcts(move, state))
    return wmcts
```

Candidate moves, obtained from ADP make the MCTS's search space smaller than before -> ADP + MCTS saves more time than the method only uses MCTS. If more than 5 -> more time consumed. If less than 5 -> less accurate results. 



