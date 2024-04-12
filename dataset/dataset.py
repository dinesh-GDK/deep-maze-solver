import os

import matplotlib.pyplot as plt
import torch
from torch.utils import data

from dataset.preprocessor import generate_dataset


class MazeData(data.Dataset):
    def __init__(self, num_mazes: int) -> None:
        self._num_mazes = num_mazes

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        maze_size, maze, solved_maze = generate_dataset()
        maze, solved_maze = torch.from_numpy(maze), torch.from_numpy(solved_maze)
        return maze_size, maze, solved_maze

    def __len__(self) -> int:
        return self._num_mazes


if __name__ == "__main__":
    os.makedirs("trash", exist_ok=True)

    for i in range(10):
        dataset = MazeData(15, 1)
        maze, solved_maze = dataset[0]
        maze = maze.squeeze().to("cpu").detach().numpy()
        solved_maze = solved_maze.squeeze().detach().numpy()

        _, ax = plt.subplots(1, 2, figsize=(100, 50))
        ax[0].imshow(maze, cmap="gray", vmin=0, vmax=1)
        ax[1].imshow(solved_maze, cmap="gray", vmin=0, vmax=1)
        ax[0].axis("off")
        ax[1].axis("off")
        plt.savefig(f"trash/data_sample_{i:04d}.png")
        plt.savefig(f"trash/output_15.png")
        plt.close()
