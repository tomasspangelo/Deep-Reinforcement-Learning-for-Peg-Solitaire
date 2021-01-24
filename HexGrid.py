import numpy as np


class HexGrid:

    def __init__(self, grid, size):
        self.size = size
        self.grid = grid

    def __str__(self):
        return "This is a HexGrid"

    def count_filled(self):
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                node = self.grid[i, j]
                if node and node.filled == True:
                    count +=1
        return count







class DiamondHexGrid(HexGrid):

    def __init__(self, size):

        HexGrid.__init__(self, DiamondHexGrid._fill_grid(size), size)

    @staticmethod
    def _fill_grid(size):
        grid = np.full((size, size), None)

        for i in range(size):
            for j in range(size):
                grid[i, j] = Node(i, j)

        for i in range(size):
            for j in range(size):
                node = grid[i, j]
                neighborhood = node.neighborhood

                if i > 0:
                    neighborhood["up"] = grid[i - 1, j]
                    if j < size - 1:
                        neighborhood["diagonal_up"] = grid[i - 1, j + 1]

                if i < size - 1:
                    neighborhood["down"] = grid[i + 1, j]
                    if j > 0:
                        neighborhood["diagonal_down"] = grid[i + 1, j - 1]
                if j < size - 1:
                    neighborhood["right"] = grid[i, j + 1]
                if j > 0:
                    neighborhood["left"] = grid[i, j - 1]
        return grid


class TriangularHexGrid(HexGrid):

    def __init__(self, size):

        HexGrid.__init__(self, TriangularHexGrid._fill_grid(size), size)

    @staticmethod
    def _fill_grid(size):
        grid = np.full((size, size), None)

        for i in range(size):
            for j in range(i+1):
                grid[i, j] = Node(i, j)

        for i in range(size):
            for j in range(i+1):
                node = grid[i, j]
                neighborhood = node.neighborhood

                if j != i:
                    neighborhood["up"] = grid[i - 1, j]
                    neighborhood["right"] = grid[i, j + 1]

                if i < size - 1:
                    neighborhood["down"] = grid[i + 1, j]
                    neighborhood["diagonal_down"] = grid[i + 1, j + 1]

                if j > 0:
                    neighborhood["left"] = grid[i, j - 1]
                    neighborhood["diagonal_up"] = grid[i - 1, j - 1]
        return grid


class Node:

    def __init__(self, row, column, filled=True):
        self.row = row
        self.column = column
        self.filled = filled
        self.neighborhood = {
            "left": None,
            "right": None,
            "up": None,
            "down": None,
            "diagonal_up": None,
            "diagonal_down": None

        }

    def is_filled(self):
        return self.filled

    def __str__(self):
        return "({row}, {column}) {filled}".format(row=self.row, column=self.column, filled=self.filled)


if __name__ == "__main__":
    d = TriangularHexGrid(4)
    a, b = (0, 0)
    print(d.grid[a,b].neighborhood)
    print(d.grid[a,b])

    for neighbor in d.grid[a, b].neighborhood:
        print(neighbor)
        print(d.grid[a, b].neighborhood[neighbor])
        print("______________________")








