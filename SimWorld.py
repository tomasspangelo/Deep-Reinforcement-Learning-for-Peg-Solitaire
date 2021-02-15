from HexGrid import DiamondHexGrid, TriangularHexGrid
import networkx as nx
import matplotlib.pyplot as plt


class SimWorld:
    """
    A class to represent a game of peg solitaire (i.e. a simulated world)
    """

    def __init__(self, open_cells, board_type='diamond', board_size=4, r_win=-50, r_loss=-50, r_step=0):
        """
        Initializes necessary variables, and sets some variables to default values.

        :param open_cells: which cells are initialized as opened.
        :param board_type: the board type, can be 'diamond' or 'triangular'
        :param board_size: the size of the board.
        :param r_win: reward for winning the game.
        :param r_loss: reward for losing the game.
        :param r_step: reward for each step of the game.
        """
        self.open_cells = open_cells
        self.finished = False
        self.goal_state = False
        self.board_type = board_type
        self.board_size = board_size

        self.visualize = False
        self.episode = []
        self.frame_delay = 0
        self.r_win = r_win
        self.r_loss = r_loss
        self.r_step = r_step

        if board_type == 'diamond':
            self.state = DiamondHexGrid(board_size)
        if board_type == 'triangular':
            self.state = TriangularHexGrid(board_size)

        self._set_empty(open_cells)
        self.remaining_pegs = self._count_remaining()

    def reset_game(self, visualize=False, frame_delay=0):
        """
        Resets the SimWorld back to initial state.

        :param visualize: True if the game should be visualized.
        :param frame_delay: Plot pauses for frame_delay seconds.
        :return: None
        """

        self.finished = False
        self.goal_state = False
        self.visualize = visualize
        self.frame_delay = frame_delay

        if self.board_type == 'diamond':
            self.state = DiamondHexGrid(self.board_size)
        if self.board_type == 'triangular':
            self.state = TriangularHexGrid(self.board_size)

        self._set_empty(self.open_cells)
        self.remaining_pegs = self._count_remaining()

        # If visualization is on, visualize the initial board.
        if visualize:
            self.graph = nx.Graph()
            self.fig = plt.figure(2)
            pos, color_map, node_size = self.state.visualize_grid(self.graph, self, None)
            nx.draw(self.graph, pos, node_color=color_map, node_size=node_size, with_labels=False,
                    font_weight='bold')
            plt.pause(5)

    def visualize_move(self, action):
        """
        Visualizes the board given an action.
        :param action: Action object that indicates the move.
        :return: None
        """
        self.graph.clear()
        plt.clf()
        pos, color_map, node_size = self.state.visualize_grid(self.graph, self, action)
        nx.draw(self.graph, pos, node_color=color_map, node_size=node_size, with_labels=False,
                font_weight='bold')
        plt.pause(self.frame_delay)

    def flatten_state(self):
        """:return: State (grid) as 1D numpy array."""
        return self.state.flatten()

    def get_legal_actions(self):
        """
        Iterates over state (grid) and finds all legal actions.
        :return: List containing legal actions (Action objects)
        """
        legal_actions = []
        current_state = self.state

        for i in range(self.board_size):
            for j in range(self.board_size):
                node = current_state.grid[i, j]
                if node and node.filled:
                    neighborhood = node.neighborhood
                    for action in neighborhood:
                        neighbor = neighborhood[action]
                        next_neighbor = neighbor.neighborhood[action] if neighbor else None
                        if neighbor and neighbor.filled and next_neighbor and not next_neighbor.filled:
                            legal_actions.append(Action(start_node=node,
                                                        jump_node=neighbor,
                                                        end_node=next_neighbor,
                                                        name=action))

        return legal_actions

    def perform_action(self, action):
        """
        Performs given action and updates state.
        :param action: Action object indicating move to be performed.
        :return: Reward for performing action.
        """
        action.jump_node.filled = False
        action.start_node.filled = False
        action.end_node.filled = True

        self.remaining_pegs -= 1
        legal_actions = len(self.get_legal_actions())

        if self.visualize:
            self.visualize_move(action)

        if legal_actions == 0 and self.remaining_pegs == 1:
            self.goal_state = True
            self.finished = True
            if self.visualize:
                self.visualize_move(None)
            return self.r_win
        if legal_actions == 0 and self.remaining_pegs > 1:
            self.finished = True
            if self.visualize:
                self.visualize_move(None)
            return self.r_loss
        return self.r_step

    def is_finished(self):
        """:return: True if game is finished, False otherwise."""
        return len(self.get_legal_actions()) == 0

    def is_goal_state(self):
        """:return: True if the current state is a goal state, False otherwise."""
        return self.goal_state

    def _count_remaining(self):
        """:return: Number of pegs"""
        return self.state.count_filled()

    def get_result(self):
        """:return: Number of remaining pegs."""
        return self.remaining_pegs

    def _set_empty(self, open_cells):
        """
        Used for initialization, setting open cells to be empty.
        :param open_cells: List of coordinates (row,column) of open cells.
        :return: None
        """
        for (i, j) in open_cells:
            self.state.grid[i, j].filled = False


class Action:
    """
    An action object represents a move in the game of peg solitaire.
    """

    def __init__(self, start_node, jump_node, end_node, name):
        """
        Initializes necessary variables
        :param start_node: Node object representing the peg that is moved.
        :param jump_node: Node object that is jumped over
        :param end_node: Node object that is moved to.
        :param name: Name of the action.
        """
        self.start_node = start_node
        self.jump_node = jump_node
        self.end_node = end_node
        self.name = name

    def __str__(self):
        """:return: String representation of Action object."""
        return "{start_node} -> {end_node}".format(start_node=self.start_node.__str__(),
                                                   end_node=self.end_node.__str__())


if __name__ == "__main__":
    pass
