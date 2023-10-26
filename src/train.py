import matplotlib.pyplot as plt
from tqdm import trange

from tic_tac_toe_environment import TicTacToeEnvironment
from agent import Agent


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


def run_n_games(
    n_games, agent_play_with,
    train=False,
    discount=.99,
    learning_rate=1.,
    eps=.2,
    plot_iter=1000,
):
    """Run a batch of games

    Args:
        n_games (int): number of games to run
        agent_play_with (str): 'x' or 'o' - agent's symbol
        train (bool, optional): whether to update agent's policy during the games. Defaults to False.
        discount (float, optional): discount factor. Defaults to .99.
        learning_rate (float, optional): learning rate used in Q-table update. Defaults to 1.
        eps (float, optional): threshold for eps-greedy exploration. Defaults to .2.
        plot_iter (int, optional): plot learning curve each `plot_iter` games. Defaults to 1000.

    Returns:
        tuple[
            TicTacToeEnvironment,
            Agent,
            dict[str, int],
            list[float],
        ]: env, trained agent, stats on wins and total rewards
    """
    env_play_with = 'o' if agent_play_with == 'x' else 'x'
    env = TicTacToeEnvironment(Agent(env_play_with))
    agent = Agent(agent_play_with, discount=discount, learning_rate=learning_rate, eps=eps)
    total_rewards = []
    stats = {
        'x': 0,
        'o': 0,
        'tie': 0,
    }

    for i in trange(n_games):
        res, total_reward = game(env, agent, train=train)
        total_rewards.append(total_reward)
        stats[res] += 1

        if i % plot_iter == 0:
            plt.bar(stats.keys(), stats.values())
            plt.title('Win Distribution');

    return env, agent, stats, total_rewards

