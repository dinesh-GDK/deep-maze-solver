import os
import random
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

BORDER = 0.0
PATH = 0.4
SOLUTION = 1.0


@dataclass
class Node:
    x: int
    y: int

    def __hash__(self) -> hash:
        return hash((self.x, self.y))

    def __add__(self, node: "Node") -> "Node":
        return Node(self.x + node.x, self.y + node.y)


class Grid:
    def __init__(self, dim_x: int = None, dim_y: int = None) -> None:
        self._grid = None
        if dim_x is not None and dim_y is not None:
            assert dim_x % 2 == 1 and dim_y % 2 == 1 and dim_x >= 3 and dim_y >= 3
            self._grid = np.ones((dim_x, dim_y)) * PATH
        self._is_maze = False

    @property
    def dim(self) -> int:
        return self._grid.shape

    @property
    def grid(self):
        return self._grid

    @property
    def isMaze(self) -> bool:
        return self._is_maze

    @isMaze.setter
    def isMaze(self, val: bool) -> bool:
        self._is_maze = val

    def get(self, x: int, y: int) -> float:
        return self._grid[x][y]

    def set(self, x: int, y: int, val: float) -> float:
        self._grid[x][y] = val
        return self.get(x, y)

    def copy(self) -> "Grid":
        new_grid = Grid()
        new_grid._grid = np.copy(self._grid)
        new_grid._is_maze = self._is_maze
        return new_grid

    def save(self, path: str) -> None:
        plt.imshow(self._grid, cmap="gray")
        plt.axis("off")
        plt.savefig(path, bbox_inches="tight")


class MazeGenerator:
    def __init__(self, grid: Grid) -> None:
        assert grid.isMaze == False
        self._maze = grid
        self._dim = grid.dim

    def _draw_border(self) -> None:
        for i in range(0, self._dim[0]):
            self._maze.set(i, 0, BORDER)
            self._maze.set(i, self._dim[1] - 1, BORDER)

        for i in range(0, self._dim[1]):
            self._maze.set(0, i, BORDER)
            self._maze.set(self._dim[0] - 1, i, BORDER)

    @staticmethod
    def _random_split(limit1: int, limit2: int, is_wall: bool = False) -> int:
        assert limit1 < limit2
        # do not include the limits
        limit1, limit2 = limit1 + 1, limit2 - 1
        split = None
        while (
            (split is None)
            or (is_wall and split % 2 == 1)
            or (not is_wall and split % 2 == 0)
        ):
            split = random.randint(limit1, limit2)
        return split

    def _recurisve_division(
        self, top_left: Node, bottom_right: Node, split_dir: bool = None
    ) -> None:
        split_dir = random.choice([True, False]) if split_dir is None else split_dir

        if bottom_right.x - top_left.x <= 2 or bottom_right.y - top_left.y <= 2:
            return

        if split_dir:
            split = MazeGenerator._random_split(top_left.x, bottom_right.x, True)
            gap = MazeGenerator._random_split(top_left.y, bottom_right.y)
            for i in range(top_left.y, bottom_right.y + 1):
                self._maze.set(split, i, BORDER)
            self._maze.set(split, gap, PATH)
            self._recurisve_division(
                top_left, Node(split, bottom_right.y), not split_dir
            )
            self._recurisve_division(
                Node(split, top_left.y), bottom_right, not split_dir
            )

        else:
            split = MazeGenerator._random_split(top_left.y, bottom_right.y, True)
            gap = MazeGenerator._random_split(top_left.x, bottom_right.x)
            for i in range(top_left.x, bottom_right.x + 1):
                self._maze.set(i, split, BORDER)
            self._maze.set(gap, split, PATH)
            self._recurisve_division(
                top_left, Node(bottom_right.x, split), not split_dir
            )
            self._recurisve_division(
                Node(top_left.x, split), bottom_right, not split_dir
            )

    def create(self) -> None:
        self._draw_border()
        # self._recurisve_division(Node(0, 0), Node(self._dim[0] - 1, self._dim[1] - 1))
        self._recurisve_division_stack()
        self._maze.isMaze = True

    def _recurisve_division_stack(self) -> None:
        split_dir = random.choice([True, False])
        stack = [(Node(0, 0), Node(self._dim[0] - 1, self._dim[1] - 1), split_dir)]

        while len(stack) > 0:
            top_left, bottom_right, split_dir = stack.pop()

            if bottom_right.x - top_left.x <= 2 or bottom_right.y - top_left.y <= 2:
                continue

            if split_dir:
                split = MazeGenerator._random_split(top_left.x, bottom_right.x, True)
                gap = MazeGenerator._random_split(top_left.y, bottom_right.y)
                for i in range(top_left.y, bottom_right.y + 1):
                    self._maze.set(split, i, BORDER)
                self._maze.set(split, gap, PATH)
                split_1 = (top_left, Node(split, bottom_right.y), not split_dir)
                split_2 = (Node(split, top_left.y), bottom_right, not split_dir)

            else:
                split = MazeGenerator._random_split(top_left.y, bottom_right.y, True)
                gap = MazeGenerator._random_split(top_left.x, bottom_right.x)
                for i in range(top_left.x, bottom_right.x + 1):
                    self._maze.set(i, split, BORDER)
                self._maze.set(gap, split, PATH)
                split_1 = (top_left, Node(bottom_right.x, split), not split_dir)
                split_2 = (Node(top_left.x, split), bottom_right, not split_dir)

            stack.append(split_1)
            stack.append(split_2)


class MazeSolver:
    def __init__(self, maze: Grid) -> None:
        assert maze.isMaze == True
        self._maze = maze
        self._dim = maze.dim

    def _recursive_dfs(self, vis: set[Node], node: Node) -> None:
        if node == Node(self._dim[0] - 2, self._dim[1] - 2):
            self._maze.set(node.x, node.y, SOLUTION)
            return True

        for diff in [Node(0, 1), Node(1, 0), Node(0, -1), Node(-1, 0)]:
            new_node = node + diff
            is_path = self._maze.get(new_node.x, new_node.y) == PATH
            inside = (
                new_node.x >= 0
                and new_node.x < self._dim[0]
                and new_node.y >= 0
                and new_node.y < self._dim[1]
            )
            if is_path and inside and new_node not in vis:
                vis.add(new_node)
                if self._recursive_dfs(vis, new_node) is True:
                    self._maze.set(node.x, node.y, SOLUTION)
                    return True

    def _stack_dfs(self) -> None:
        dirs = [Node(0, 1), Node(1, 0), Node(0, -1), Node(-1, 0)]
        begin = Node(1, 1)
        destination = Node(self._dim[0] - 2, self._dim[1] - 2)

        vis = {begin}
        stack = [begin]
        parent_mapper = {begin: None}

        while len(stack) > 0:
            node = stack.pop(0)
            if node == destination:
                break

            for diff in dirs:
                new_point = node + diff
                inside = (
                    new_point.x >= 0
                    and new_point.x < self._dim[0]
                    and new_point.y >= 0
                    and new_point.y < self._dim[1]
                )
                if inside and new_point not in vis:
                    vis.add(new_point)
                    is_path = self._maze.get(new_point.x, new_point.y) == PATH
                    if is_path:
                        parent_mapper[new_point] = node
                        stack.append(new_point)

        while node in parent_mapper:
            self._maze.set(node.x, node.y, SOLUTION)
            node = parent_mapper[node]

    def solve(self) -> None:
        # vis = {Node(1, 1)}
        # self._recursive_dfs(vis, Node(1, 1))
        self._stack_dfs()


if __name__ == "__main__":
    tic = datetime.now()
    os.makedirs("trash", exist_ok=True)
    for i in range(1):
        maze = Grid(57, 57)
        MazeGenerator(maze).create()

        solved_maze = maze.copy()
        MazeSolver(solved_maze).solve()

        maze = maze.grid
        solved_maze = solved_maze.grid

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(maze, cmap="gray", vmin=0, vmax=1)
        ax[1].imshow(solved_maze, cmap="gray", vmin=0, vmax=1)
        ax[0].axis("off")
        ax[1].axis("off")
        plt.savefig(f"trash/sample_{i:04d}.png", bbox_inches="tight")
        plt.close()

    print(f"Time: {datetime.now() - tic}")
