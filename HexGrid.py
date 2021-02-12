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
                    count += 1
        return count


class DiamondHexGrid(HexGrid):

    def __init__(self, size):

        HexGrid.__init__(self, DiamondHexGrid._fill_grid(size), size)

    def flatten(self):
        return np.array([1 if node.filled else 0 for node in np.concatenate(self.grid)])

    @staticmethod
    def _fill_grid(size):
        grid = np.full((size, size), None)

        for i in range(size):
            for j in range(size):
                node = Node(i, j)
                grid[i, j] = node

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

    @staticmethod
    def visualize_grid(graph, state, action=None):
        pos = {}
        total_height = 1 + 2 * state.board_size
        total_width = total_height

        color_map = []
        size_map = []

        if action:
            start_node = action.start_node
            end_node = action.end_node
            jump_node = action.jump_node

        grid = state.state.grid
        for i in range(state.board_size):
            h = total_height - i
            w = np.ceil(total_width / 2) - i
            for j in range(state.board_size):
                node = grid[i, j]
                if node:
                    graph.add_node(node)
                    pos[node] = (w, h)
                    if node.filled:
                        color_map.append('blue')
                    elif action and node.__str__() == jump_node.__str__():
                        color_map.append('red')
                    else:
                        color_map.append('grey')

                    if action and node.__str__() == start_node.__str__():
                        size_map.append(200)

                    elif action and node.__str__() == end_node.__str__():
                        size_map.append(800)

                    else:
                        size_map.append(500)
                    h -= 1
                    w += 1

        for i in range(state.board_size):
            for j in range(state.board_size):
                node = grid[i, j]
                if node:
                    for action in node.neighborhood:
                        neighbor = node.neighborhood[action]
                        if neighbor:
                            graph.add_edge(node, neighbor)
        return pos, color_map, size_map


class TriangularHexGrid(HexGrid):

    def __init__(self, size):

        HexGrid.__init__(self, TriangularHexGrid._fill_grid(size), size)

    def flatten(self):
        flat = []
        for i in range(self.size):
            for j in range(i + 1):
                flat.append(1 if self.grid[i, j].filled else 0)

        return np.array(flat)

    @staticmethod
    def _fill_grid(size):
        grid = np.full((size, size), None)

        for i in range(size):
            for j in range(i + 1):
                node = Node(i, j)
                grid[i, j] = node

        for i in range(size):
            for j in range(i + 1):
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

    @staticmethod
    def visualize_grid(graph, state, action=None):
        pos = {}
        total_height = state.board_size
        total_width = total_height * 2 - 1

        color_map = []
        size_map = []

        if action:
            start_node = action.start_node
            end_node = action.end_node
            jump_node = action.jump_node

        grid = state.state.grid
        for i in range(state.board_size):
            h = total_height - i
            w = np.ceil(total_width / 2) - i
            for j in range(i + 1):
                node = grid[i, j]
                if node:
                    graph.add_node(node)
                    pos[node] = (w, h)
                    if node.filled:
                        color_map.append('blue')
                    elif action and node.__str__() == jump_node.__str__():
                        color_map.append('red')
                    else:
                        color_map.append('grey')

                    if action and node.__str__() == start_node.__str__():
                        size_map.append(200)

                    elif action and node.__str__() == end_node.__str__():
                        size_map.append(800)

                    else:
                        size_map.append(500)

                    w += 2

        for i in range(state.board_size):
            for j in range(i + 1):
                node = grid[i, j]
                if node:
                    for action in node.neighborhood:
                        neighbor = node.neighborhood[action]
                        if neighbor:
                            graph.add_edge(node, neighbor)
        return pos, color_map, size_map


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
        return "({row}, {column})".format(row=self.row, column=self.column, filled=self.filled)


if __name__ == "__main__":
    '''
    d = DiamondHexGrid(4)
    a, b = (0, 0)
    print(d.grid[a,b].neighborhood)
    print(d.grid[a,b])

    d.grid[1,1].filled=False

    for neighbor in d.grid[a, b].neighborhood:
        print(neighbor)
        print(d.grid[a, b].neighborhood[neighbor])
        print("______________________")

    print(d.flatten())
    '''
