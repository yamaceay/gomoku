AlphaZero Gomoku

Bulletpoints:

* Generalize the AlphaZero approach for the Gomoku game and achieve impressive results
* Initiate from a state of random play, and without any domain knowledge apart from the game rules, learn a winning strategy on a 6x6 table after just a few hours of training on an economical GPU.
* Embark on an extensive research endeavor, and juxtapose the efficacy of our refined AlphaZero methodology against a conventional method that exclusively leverages the MCTS.
* Critically assess how these two distinct techniques fare in terms of both efficiency and effectiveness under comparable conditions, to shed light on their relative strengths and potential areas of improvement.

Deep Learning:
* Value Network (V): Estimate the value ( = the expected outcome) of a given state. Values close to +1 indicate favorable outcomes for the player and values close to -1 indicate unfavorable outcomes. 
* Policy Network ($\pi$): Provide a probability distribution over all possible moves from a given state. For state s and an action a, $\pi(a | s)$ represents the probability of taking action a as a highly optimized game player.

Monte Carlo Tree Search (MCTS):
* Policy Network: serves as the beacon guiding the expansion of the search tree. Instead of branching out indiscriminately, it casts the spotlight on moves radiating promise and potential, ensuring that the exploration process remains strategic and focused.
* Value Network: Adept evaluator, meticulously scrutinizing leaf nodes within the tree. It diminishes the traditional reliance on random rollouts for evaluation, infusing the processs with a heightened level of precision. It not only speeds up the evaluation but also endows it with a more profound insight into the game's dynamics. 



