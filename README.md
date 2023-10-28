# RL Agents Learning to Play TicTacToe 3x3 using Q-Learning

## Introduction

In this project, we aim to train RL agents to play 3x3 tic-tac-toe using tabular Q-learning algorithm. The agents will learn to make optimal moves based on past experiences and rewards obtained from the game.

## Game rules
TicTacToe is a two-player game played on a 3x3 grid. The objective is to form a straight line (horizontally, vertically, or diagonally) with three of your own markers (either 'X' or 'O').

## Tabular Q-learning

Tabular Q-learning is a reinforcement learning method that involves using a tabular representation of the Q-values to learn an optimal policy for decision-making in an environment. It is suitable for tasks where state and action space is discrete and small enough to be represented as a table.
Q-learning employs a Q-table, which is a tabular representation storing the Q-values for all possible state-action pairs. The number of rows in the table corresponds to the number of unique states in the environment, and the number of columns represents the number of available actions. Q-learning updates the Q-values iteratively based on the Bellman equation, which states that the optimal Q-value for a state-action pair is the expected immediate reward plus the discounted maximum Q-value of the next state. This process continues until the Q-values converge to their optimal values.

The Q-values are updated according to the formula: 
$$Q_{new}(s_t, a_t) = Q(s_t, a_t) + \alpha \cdot [R(s_t, a_t) + \gamma \cdot max_a Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

where Q<sub>new</sub>(s<sub>t</sub>, a<sub>t</sub>) is the old Q-value of taking action a in state s, R is instantaneous reward, and Q(s<sub>t+1</sub>, a+1<sub>t</sub>) the maximum Q-value we can get for our next move after taking action a and ending up in state s+1.

To balance exploration and exploitation, we use epsilon-greedy strategy, where sometimes instead of picking the optimal move from the Q table we take a random move with chance 𝜖.

## Implementation Details

We implement the Q-learning algorithm specific to the problem as follows:

1. **Environment**

| Accepts        | Returns            |
|----------------|--------------------|
| Action         | State, reward      |

The state of the game is represented as a 3x3 numpy array, where each element can be 1 (cell is occupied by the player agent), -1 (cell is occupied by the rival agent) or 0 (cell is empty).
Implements general mechanics of the game and checks winning conditions.

2. **Model**

| Accepts                                                                         | Returns            |
|---------------------------------------------------------------------------------|--------------------|
| play_with 'x' or 'o', discount factor, learning_rate, epsilon| Action             |

Implements the tabular Q-learning algorithm.

3. **Training**

Runs a batch of game sessions to train the agent. A single game lasts until someone wins or players draw.

## Training the Model

To train the RL agents, we follow these steps:

1. Initialize the Q-table with zero values for all state-action pairs.
2. Train the newly created Q-agent play as x against random rival.
3. Duplicate the trained agent as a rival.
4. Create another agent as in pt. 1
5. Train the newly created Q-agent play as o against the trained Q-agent.

## Rewards

We assign the rewards as shown in the table below:

| Outcome            | Reward             |
|--------------------|--------------------|
| Agent wins         | + 10               |
| Agent loses        | - 10               |
| Draw               | + 5                |

Assigning a positive reward for draw is potentially beneficial for an 'o' player as in case with playing against a very well trained 'x' agent draw is the best possible outcome.

## Results and Discussion

- Win rate against random players.
- Win rates between Q-agents.

Both the trained agent playing as x and the trained agent playing as o show good results versus random player. However, trained x almost always wins with trained o. In our opinion, this discrepancy arises from the rules and dynamics of the game. The 'x' player typically gets to make the first move in tic-tac-toe. By playing first, it has the opportunity to control the initial placement on the game board and set the tone for the rest of the game, while the main goal for the 'o' player who normally moves second is forced to defend and respond rather than focusing solely on his own strategy.

When analyzing tic-tac-toe mathematically, researchers have determined that if both players play optimally, assuming no mistakes, the 'x' player can always force a draw or win in the best-case scenario. This effect of the mathematically proven advantage is exactly what can be observed in our simulations. It is important to mention that while the 'x' player generally has an advantage, in real life, skilled players can still overcome this and win as the 'o' player, because there is always a human factor. Based on our experience of human vs trained 'o' Q-player matches, it performs pretty well. Trust us, it's smarter than you think (you can try it yourself to make sure).

## Conclusion

In this project, we successfully trained RL agents to play TicTacToe on a 3x3 board using the Q-learning algorithm. The trained agents demonstrated the ability to make informed decisions based on past experiences and rewards. We observed that the trained x player almost always wins with trained o. The results show decent performance, but further improvements and exploration can be done to enhance the agent's capabilities, especially the possibilities for the 'o' agent to beat 'x'.
