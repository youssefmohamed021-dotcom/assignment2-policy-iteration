from grid_maze_env import GridMazeEnv
from policy_iteration import policy_iteration
import time


layout = {
    "agent": (4, 0),
    "goal": (0, 4),
    "bad1": (1, 1),
    "bad2": (2, 3)
}

policy, values, iterations = policy_iteration(
    grid_size=5,
    goal_pos=layout["goal"],
    bad_cells=[layout["bad1"], layout["bad2"]]
)

print("Policy Iteration converged after:", iterations, "iterations")

env = GridMazeEnv(render_mode="human")
observation, info = env.reset(options={"layout": layout})

total_reward = 0.0

for step in range(50):
    row, col = observation[0], observation[1]
    state = row * 5 + col
    action = policy[state]

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    print(
        "Step:", step + 1,
        "Action:", action,
        "Reward:", reward,
        "Position:", tuple(observation[:2])
    )

    time.sleep(0.4)

    if terminated or truncated:
        break

print("Final reward:", total_reward)
env.close()