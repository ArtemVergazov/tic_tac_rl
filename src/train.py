from tic_tac_toe_environment import TicTacToeEnvironment
from agent import Agent
import matplotlib.pyplot as plt


def game(env, agent, train=False):
    """
    Run a single game until someone wins or players draw. If the board is not empty, cleans it.

    Args:
        env (TicTacToeEnvironment): clean environment
        agent (Agent): plays with the environment
        train (bool): whether to update agent's policy in the process

    Returns:
        str: winner symbol or 'tie' if tie
    """
    env.reset()
    next_state = env.board.copy()
    
    total_reward = 0

    if env.trained_player.play_with == 'x':
        env.strategy_step()

    while True:
        state = next_state.copy()
        action = agent.sample_action(state)
        next_state, reward, winner, tie = env.step(action)
        total_reward += reward
        
        if train:
            agent.update_policy(state, action, next_state, reward)

        if winner:
            return winner, total_reward
        if tie:
            return 'tie', total_reward


def run_n_games(n_games, agent_play_with, train=False, discount=.99, learning_rate=1., eps=.2):
    env_play_with = 'o' if agent_play_with == 'x' else 'x'
    env = TicTacToeEnvironment(Agent(env_play_with))
    agent = Agent(agent_play_with, discount=discount, learning_rate=learning_rate, eps=eps)
    total_rewards = []
    stats = {
        'x': 0,
        'o': 0,
        'tie': 0,
    }

    for _ in range(n_games):
        res, total_reward = game(env, agent, train=train)
        total_rewards.append(total_reward)
        stats[res] += 1

    plt.plot(total_rewards, 'o')
    plt.xlabel('Game num')
    plt.ylabel('Game reward')
    plt.title('Learning Curve')
    return env, agent, stats

