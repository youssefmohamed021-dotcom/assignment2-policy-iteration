from gymnasium.wrappers import RecordVideo
from grid_maze_env import GridMazeEnv
from policy_iteration import policy_iteration


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

env = RecordVideo(
    GridMazeEnv(render_mode="rgb_array"),
    video_folder="videos",
    episode_trigger=lambda episode: True,
    name_prefix="policy_iteration"
)

observation, info = env.reset(options={"layout": layout})

for step in range(50):
    row, col = observation[0], observation[1]
    state = row * 5 + col
    action = policy[state]

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()

print("Video created in the videos folder.")