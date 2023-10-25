import random
import numpy as np


class Agent:
    def __init__(self, play_with, discount=.99, learning_rate=.1, eps=.05):
        """Create agent

        Args:
            play_with (str): agent plays with either 'x' or 'o'
            discount (float): discount factor
            learning_rate (float): parameter used in updating Q-table
            eps (float): parameter for eps-greedy exploration
        """
        assert play_with in {'x', 'o'}, 'Agent symbol must be "x" or "o"'
        self.play_with = play_with
        self.qtable = {}
        self.discount = discount
        self.learning_rate = learning_rate
        self.eps = eps

    def sample_action(self, state):
        """Sample action given current state according to current Q-table

        Args:
            state (np.ndarray): state of the board

        Returns:
            list: sampled action as (int, int) pair
        """
        hashable_state = self.hashable_state(state)

        # If state does not exist in the Q-table, insert it
        if hashable_state not in self.qtable:
            self.init_qtable_entry(state)

        available_positions = self.action_space(state)

        # eps-greedy
        if random.uniform(0, 1) < self.eps:
            return random.choice(available_positions)
        return self.optimal_action(hashable_state)

    def optimal_action(self, hashable_state):
        """Get action with the highest Q-value
        If there are multiple such actions, sample one of them randomly

        Args:
            hashable_state (tuple[str]): current state

        Returns:
            list: chosen action as an (int, int) pair
        """
        actions = list(self.qtable[hashable_state].keys())
        values = list(self.qtable[hashable_state].values())

        # If there are multiple max values, choose randomly for the sake of exploration
        assert np.isfinite(values).all(), f'{values[~np.isfinite(values)]} in values at indices {np.argwhere(~np.isfinite(values))}'
        return actions[int(random.choice(np.argwhere(values == np.max(values))))]

    def update_policy(self, state, action, next_state, reward):
        """Update action sampling policy based on state, sampled action and obtained reward

        Args:
            state (np.ndarray): state of the board
            action (list): sampled action as (int, int) pair
            next_state (np.ndarray): state after taking the action
            reward (float): reward obtained from state-action pair
        """
        hashable_state = self.hashable_state(state)
        
        # If state does not exist in the Q-table, insert it
        if hashable_state not in self.qtable:
            self.init_qtable_entry(state)

        # Update the Q-table
        old_value = self.qtable[hashable_state][action]

        hashable_next_state = self.hashable_state(next_state)
        if hashable_next_state in self.qtable:
            next_max = max(self.qtable[hashable_next_state].values())
        else:
            next_max = 0

        new_value = old_value + self.learning_rate*(reward+self.discount*next_max-old_value)
        self.qtable[hashable_state][action] = new_value
    
    def init_qtable_entry(self, state):
        """Initialize Q-table entry for the given state with zeros

        Args:
            state (np.ndarray): state of the board
        """
        hashable_state = self.hashable_state(state)
        available_actions = self.action_space(state)
        self.qtable[hashable_state] = {action: 0 for action in available_actions}
        

    @staticmethod
    def action_space(state):
        """Obtain available actions given current state

        Args:
            state (np.ndarray): state of the board

        Returns:
            list: available position as a list of (int, int) pairs
        """
        x, y = np.where(state == '')
        return [(x_, y_) for x_, y_ in zip(x, y)]

    @staticmethod
    def hashable_state(state):
        """Q-table should work with Python lists rather than np.ndarray-s for hashing reasons.
        We also reshape list of lists to a single list for simplicity.

        Args:
            state (np.ndarray): state coming from the board

        Returns:
            _type_: _description_
        """
        return tuple(state.reshape((-1,)))

