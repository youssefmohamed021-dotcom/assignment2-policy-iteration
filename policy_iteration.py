import numpy as np


def state_to_position(state, grid_size):
    return state // grid_size, state % grid_size


def position_to_state(position, grid_size):
    return position[0] * grid_size + position[1]


def get_transition(state, action, grid_size, goal_pos, bad_cells):
    row, col = state_to_position(state, grid_size)

    if action == 0:
        col += 1
    elif action == 1:
        row -= 1
    elif action == 2:
        col -= 1
    elif action == 3:
        row += 1

    row = int(np.clip(row, 0, grid_size - 1))
    col = int(np.clip(col, 0, grid_size - 1))

    next_position = (row, col)
    next_state = position_to_state(next_position, grid_size)

    if next_position == goal_pos:
        return next_state, 1.0, True

    if next_position in bad_cells:
        return next_state, -1.0, True

    return next_state, 0.0, False


def policy_evaluation(policy, grid_size, goal_pos, bad_cells, gamma=0.95, theta=1e-8):
    n_states = grid_size * grid_size
    values = np.zeros(n_states)
    terminal_positions = set(bad_cells + [goal_pos])

    while True:
        delta = 0.0

        for state in range(n_states):
            position = state_to_position(state, grid_size)

            if position in terminal_positions:
                continue

            old_value = values[state]
            action = policy[state]
            next_state, reward, terminated = get_transition(
                state,
                action,
                grid_size,
                goal_pos,
                bad_cells
            )

            if terminated:
                values[state] = reward
            else:
                values[state] = reward + gamma * values[next_state]

            delta = max(delta, abs(old_value - values[state]))

        if delta < theta:
            break

    return values


def policy_iteration(grid_size, goal_pos, bad_cells, gamma=0.95):
    n_states = grid_size * grid_size
    policy = np.zeros(n_states, dtype=int)
    terminal_positions = set(bad_cells + [goal_pos])
    iterations = 0

    while True:
        iterations += 1

        values = policy_evaluation(
            policy,
            grid_size,
            goal_pos,
            bad_cells,
            gamma
        )

        stable = True

        for state in range(n_states):
            position = state_to_position(state, grid_size)

            if position in terminal_positions:
                continue

            old_action = policy[state]
            action_values = np.zeros(4)

            for action in range(4):
                next_state, reward, terminated = get_transition(
                    state,
                    action,
                    grid_size,
                    goal_pos,
                    bad_cells
                )

                if terminated:
                    action_values[action] = reward
                else:
                    action_values[action] = reward + gamma * values[next_state]

            policy[state] = int(np.argmax(action_values))

            if policy[state] != old_action:
                stable = False

        if stable:
            break

    return policy, values, iterations