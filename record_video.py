from gymnasium.wrappers import RecordVideo
from grid_maze_env import GridMazeEnv
from policy_iteration import policy_iteration


layouts = [
    {
        "agent": (4, 0),
        "goal": (0, 4),
        "bad1": (1, 1),
        "bad2": (2, 3)
    },
    {
        "agent": (0, 0),
        "goal": (4, 4),
        "bad1": (1, 2),
        "bad2": (3, 1)
    },
    {
        "agent": (4, 4),
        "goal": (0, 0),
        "bad1": (2, 2),
        "bad2": (1, 3)
    }
]


for index, layout in enumerate(layouts, start=1):
    policy, values, iterations = policy_iteration(
        grid_size=5,
        goal_pos=layout["goal"],
        bad_cells=[layout["bad1"], layout["bad2"]]
    )

    env = RecordVideo(
        GridMazeEnv(render_mode="rgb_array"),
        video_folder="videos",
        episode_trigger=lambda episode: True,
        name_prefix=f"policy_iteration_maze_{index}"
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

    print(
        f"Maze {index}: {iterations} Policy Iteration rounds, "
        f"final reward = {reward}"
    )

print("Three videos were created in the videos folder.")