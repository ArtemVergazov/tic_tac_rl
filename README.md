# RL Agents Learning to Play TicTacToe 3x3 using Q-Learning

## Introduction

In this project, we aim to train RL agents to play 3x3 tic-tac-toe using tabular Q-learning algorithm. The agents will learn to make optimal moves based on past experiences and rewards obtained from the game.

## Setup instruction
Repository comes with two requirements files: [requirements.txt](requirements.txt) (necessary for training the model and verifying results) and [requirements_recommended.txt](requirements_recommended.txt) (needed for running the minigame with GUI which allows to play with our pre-trained models or models you will want to train on your own.). The minigame runs on PySide6, so that's basically all the difference.

### minimal setup (train/test)
Install the dependencies (you might wanna setup a `venv` or a `conda env` first but we assume you are comfortable with doing that).

```
cd tic_tac_rl
pip install -r requirements.txt
```

If you want to make improvements to the architecture, edit file in `tictac_rl`. To recreate our experiments, or try something of your own, use `train.ipynb`, where the train/test pipelines are.

### setup with the minigame
Install the dependencies.

```
cd tic_tac_rl
pip install -r requirements_recommended.txt
```

Launch the game:
```
python app.py
```

## Game rules
TicTacToe is a two-player game played on a 3x3 grid. The objective is to form a straight line (horizontally, vertically, or diagonally) with three of your own markers (either 'X' or 'O').

## Tabular Q-learning

Tabular Q-learning is a reinforcement learning method that involves using a tabular representation of the Q-values to learn an optimal policy for decision-making in an environment. It is suitable for tasks where state and action space is discrete and small enough to be represented as a table.
Q-learning employs a Q-table, which is a tabular representation storing the Q-values for all possible state-action pairs. The number of rows in the table corresponds to the number of unique states in the environment, and the number of columns represents the number of available actions. Q-learning updates the Q-values iteratively based on the Bellman equation, which states that the optimal Q-value for a state-action pair is the expected immediate reward plus the discounted maximum Q-value of the next state. This process continues until the Q-values converge to their optimal values.

The Q-values are updated according to the formula: 
$$Q_{new}(s_t, a_t) = Q(s_t, a_t) + \alpha \cdot [R(s_t, a_t) + \gamma \cdot max_a Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

where Q<sub>new</sub>(s<sub>t</sub>, a<sub>t</sub>) is the old Q-value of taking action a in state s, R is instantaneous reward, and Q(s<sub>t+1</sub>, a<sub>t+1</sub>) the maximum Q-value we can get for our next move after taking action a and ending up in state s+1.

To balance exploration and exploitation, we use epsilon-greedy strategy, where sometimes instead of picking the optimal move from the Q table we take a random move with chance 𝜖.

## Implementation Details

We implement the Q-learning algorithm specific to the problem as follows:

1. **Environment**

| Accepts        | Returns            |
|----------------|--------------------|
| Action         | State, reward      |

The state of the game is represented as a 3x3 numpy array, where each element can be 'x', 'o' or '' (cell is empty).
Implements general mechanics of the game and checks winning conditions.

> Note on the agent-to-env relationship.
> Environment object has an Agent instance as its member.
> Such arcitecture allows us to have agent-to-agent dynamics:
> while agent interacts with the environment (which allows for all the power of RL framework),
> in reality, it interacts with another instance of Agent, and,
> in fact, it is two agents playing with each other.
> Environment class is used to:
> - accept input from the "outside" Agent
> - transfer it to the agent that is its member
> - accept response from the member agent
> - and return it as Environment's next state

> Also, Environment is responsible for reward.

2. **Agent**

| Accepts                                                                         | Returns            |
|---------------------------------------------------------------------------------|--------------------|
| play_with 'x' or 'o', discount factor, learning_rate, epsilon| Action             |

Implements the tabular Q-learning algorithm.

3. **Training**

Runs a batch of game sessions to train the agent. A single game lasts until someone wins or players draw.

## Training the Model

To train the RL 'x' and 'o' agents, we follow these steps:

1. Initialize the Q-table with zero values for all state-action pairs for both agents.
2. Record 'x' win rate vs. 'o' rival (will be used as metric).
3. Train the newly created Q-agent playing as 'x' against "fixed" random 'o' rival for `n_games`.
4. Now, treat the trained 'x' agent as part of the environment.
5. Record 'o' win rate vs. random rival (will be used as metric).
5. Train the 'o' agent vs. 'x' rival for `n_games`.

Repeat steps 2-5 for `num_epoch`.

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

Both the trained agent playing as x and the trained agent playing as o show good results versus random player. However, trained x almost always wins with trained o. In our opinion, this discrepancy arises from the rules and dynamics of the game. The 'x' player typically gets to make the first move in tic-tac-toe. By playing first, it has the opportunity to control the initial placement on the game board and set the tone for the rest of the game, while the main goal for the 'o' player who normally moves second is forced to defend and respond rather than focusing solely on its own strategy.

When analyzing tic-tac-toe mathematically, researchers have determined that if both players play optimally, assuming no mistakes, the 'x' player can always force a draw or win in the best-case scenario. This effect of the mathematically proven advantage is exactly what can be observed in our simulations. It is important to mention that while the 'x' player generally has an advantage, in real life, skilled players can still overcome this and win as the 'o' player, because there is always a human factor. Based on our experience of human vs trained 'o' Q-player matches, it performs pretty well. Trust us, it's smarter than you think (you can try it yourself to make sure).

## Conclusion

In this project, we successfully trained RL agents to play TicTacToe on a 3x3 board using the Q-learning algorithm. The trained agents demonstrated the ability to make informed decisions based on past experiences and rewards. We observed that the trained x player almost always wins with trained o. The results show decent performance, but further improvements and exploration can be done to enhance the agent's capabilities, especially the possibilities for the 'o' agent to beat 'x'.
