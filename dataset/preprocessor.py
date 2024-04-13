import random

import cv2
import numpy as np

from constants import MAZE_MAX_SIZE, MAZE_MIN_SIZE, MODEL_INPUT_SIZE
from dataset.maze import Grid, MazeGenerator, MazeSolver


def generate_dataset(maze_size: int = None) -> tuple[int, np.ndarray, np.ndarray]:
    if maze_size is None:
        maze_size = random.randint(MAZE_MIN_SIZE // 2, MAZE_MAX_SIZE // 2) * 2 + 1

    maze = Grid(maze_size, maze_size)
    MazeGenerator(maze).create()

    solved_maze = maze.copy()

    MazeSolver(solved_maze).solve()

    maze = maze.grid
    maze[maze > 0] = 1

    solved_maze = solved_maze.grid
    solved_maze[solved_maze < 1] = 0

    maze = cv2.resize(
        maze, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_NEAREST
    )
    solved_maze = cv2.resize(
        solved_maze,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    maze = np.expand_dims(maze, axis=0).astype(np.float32)
    solved_maze = np.expand_dims(solved_maze, axis=0).astype(np.float32)

    return maze_size, maze, solved_maze
