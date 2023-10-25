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
    next_state = env.board

    while True:
        state = next_state
        action = agent.sample_action(state)
        next_state, reward, winner, tie = env.step(action)
        
        if train:
            agent.update_policy(state, action, next_state, reward)

        if winner:
            return winner
        if tie:
            return 'tie'

