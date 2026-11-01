# Assignment 2 Policy Iteration Grid Maze

This project implements a custom 5x5 Grid Maze Gymnasium environment and solves it using Policy Iteration.

## Features

- Custom Gymnasium environment
- 5x5 grid maze
- Random agent, goal, and two bad-cell positions
- PyGame visual rendering
- Policy Iteration algorithm
- MP4 video recording using RecordVideo

## Actions

- 0: Right
- 1: Up
- 2: Left
- 3: Down

## Rewards

- +1 for reaching the goal
- -1 for reaching a bad cell
- 0 otherwise

## Run

Install packages:

```bash
pip install -r requirements.txt
```

Run the Policy Iteration agent:

```bash
python main.py
```

Create an MP4 video:

```bash
python record_video.py
```