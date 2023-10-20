import numpy as np


class TicTacToeEnvironment:
    """RL environment - accepts action, returns state, reward.
    Environment has a specific strategy, or policy, not to confuse with agent's policy.
    """
    def __init__(self, trained_player):
        """Create Tic Tac Toe environment

        Args:
            env_play_with (str): symbol environment plays with, either 'x' or 'o'
            trained_player: trained agent that makes moves for environment
        """

        # 5x5 board filled with '', 'x' or 'o'
        self.reset()
        self.trained_player = trained_player
        self.agent_play_with = 'x' if self.trained_player.play_with == 'o' else 'o'

        if self.trained_player.play_with == 'x':  # 'x' go first
            self.strategy_step()

    def reset(self):
        """Clean the board
        """
        self.board = np.zeros((5, 5), dtype=str)

    def step(self, player_position):
        """Receive action and return next state, reward

        Args:
            player_position (list): (int, int) pair - action that comes from the agent
                assuming action is bounded correctly

        Returns:
            tuple[np.ndarray, float, str | None]: encoded numerical state, reward and
                winner symbol if the game ends
        """

        reward = self.get_reward(player_position)

        self.board[*player_position] = self.agent_play_with
        winner = self.check_winner()
        tie = self.check_tie()
        if winner != self.agent_play_with:
            if not tie:
                self.strategy_step()
                winner = self.check_winner()
                tie = self.check_tie()

        return self.board2state(), reward, winner, tie
    
    def board2state(self):
        """Encode board to numeric state

        agent's symbol - 1, environment's symbol - -1, empty cell - 0

        Returns:
            np.ndarray: encoded numerical state as 5x5 int array
        """
        agent_mask = np.where(self.board == self.agent_play_with, 1, 0)
        env_mask = np.where(self.board == self.trained_player.play_with, -1, 0)
        return agent_mask + env_mask
    
    def get_reward(self, action):
        """Reward for RL agent based on current state and its action

        Args:
            action (list): agent's action as (int, int) pair
        
        Returns:
            float: reward
        """
        state = self.board2state()        
        
        # Randomly as of now
        reward = 1.

        if self.check_winner() == self.agent_play_with:
            reward += 50

        return reward

    def strategy_step(self):
        """Feed state to the environment's trained player and sample action.
        Apply sampled action to the board.
        """
        state = self.board2state()
        position = self.trained_player.sample_action(state)
        self.board[*position] = self.trained_player.play_with
    
    def check_winner(self):
        """Check winner of the game, return winner symbol

        Returns:
            str | None: winner symbol or None if still playing
        """
        for symbol in 'xo':
            if (
                # Check horizontal
                (self.board[:, :-1] == symbol).all(axis=1).any() or
                (self.board[:, 1:] == symbol).all(axis=1).any() or
                # Check vertical
                (self.board[:-1, :] == symbol).all(axis=0).any() or
                (self.board[1:, :] == symbol).all(axis=0).any() or
                # Check diagonals
                (self.board[range(4), range(4)] == symbol).all() or
                (self.board[range(1, 5), range(1, 5)] == symbol).all() or
                (self.board[range(4), range(4, 0, -1)] == symbol).all() or
                (self.board[range(1, 5), range(3, -1, -1)] == symbol).all() or
                # Check subdiagonals
                (self.board[range(4), range(1, 5)] == symbol).all() or
                (self.board[range(1, 5), range(4)] == symbol).all() or
                (self.board[range(4), range(3, -1, -1)] == symbol).all() or
                (self.board[range(1, 5), range(4, 0, -1)] == symbol).all()
            ):
                return symbol
    
    def check_tie(self):
        """Checks if the players draw

        Returns:
            bool: whether it is a tie
        """
        return (self.board != '').all()

