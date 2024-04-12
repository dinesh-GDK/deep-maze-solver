import argparse

import matplotlib.pyplot as plt
import torch

from constants import MAZE_MAX_SIZE, MAZE_MIN_SIZE
from dataset.dataset import generate_dataset


def maze_size_type(x) -> int | None:
    if x is None:
        return x
    x = int(x)
    if not (x >= MAZE_MIN_SIZE and x <= MAZE_MAX_SIZE and x % 2 == 1):
        raise argparse.ArgumentTypeError(
            f"Invalid maze size; Size should be between {MAZE_MIN_SIZE} and {MAZE_MAX_SIZE}, and a odd number"
        )
    return x


def main(maze_size: int, model_path: str, output_path: str) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maze_size, maze, solved_maze = generate_dataset(maze_size)
    maze, solved_maze = torch.from_numpy(maze), torch.from_numpy(solved_maze)

    model = torch.load(model_path)
    model.to(device)
    model.eval()

    maze = maze.unsqueeze(0)
    maze = maze.to(device)
    predict = model(maze).squeeze()
    solved_maze = solved_maze.squeeze().detach().numpy()
    predict = predict.to("cpu").detach().numpy()
    maze = maze.squeeze().to("cpu").detach().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(maze, cmap="gray", vmin=0, vmax=1)
    ax[1].imshow(solved_maze, cmap="gray", vmin=0, vmax=1)
    ax[2].imshow(predict, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Maze")
    ax[1].set_title("Solution")
    ax[2].set_title("Model Prediction")
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    fig.text(0.5, 0.1, f"Maze Dim: {maze_size}x{maze_size}", ha="center")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maze Inference")
    parser.add_argument(
        "-s", "--maze-size", type=maze_size_type, help="size of the maze"
    )
    parser.add_argument("-m", "--model", type=str, help="path to pytorch model")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.png",
        help="output path for the image",
    )
    args = parser.parse_args()
    main(args.maze_size, args.model, args.output)

# python3 inference.py -s 51 -m Results/DEBUG/models/best_train_model.pt -o output.py
