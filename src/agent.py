import random
import numpy as np


class Agent:
    def __init__(self, play_with):
        """Create agent

        Args:
            play_with (str): agent plays with either 'x' or 'o'
        """
        assert play_with in {'x', 'o'}, 'Agent symbol must be "x" or "o"'
        self.play_with = play_with
        self.qtable = {}

    @staticmethod
    def action_space(state):
        """Obtain available actions given current state

        Args:
            state (np.ndarray): state of the board

        Returns:
            list: available position as a list of (int, int) pairs
        """
        x, y = np.where(state == 0)
        return [(x_, y_) for x_, y_ in zip(x, y)]

    def sample_action(self, state):
        """Sample action given current state according to some policy

        Args:
            state (np.ndarray): state of the board encoded as
            1 - agent's symbol,
            -1 - environment's symbol,
            0 - empty cell

        Returns:
            list: sampled action as (int, int) pair
        """
        available_positions = self.action_space(state)

        # Random as of now
        position = random.choice(available_positions)

        return position

    def update_policy(self, state, action, reward):
        """Update action sampling policy based on state, sampled action and obtained reward

        Args:
            state (np.ndarray): state of the board
            action (list): sampled action as (int, int) pair
            reward (float): reward obtained from state-action pair
        """
        pass

