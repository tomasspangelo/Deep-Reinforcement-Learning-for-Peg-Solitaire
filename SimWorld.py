from HexGrid import DiamondHexGrid, TriangularHexGrid
import networkx as nx
import matplotlib.pyplot as plt

import copy


class SimWorld:

    def __init__(self, open_cells, board_type='diamond', board_size=4):
        self.finished = False
        self.goal_state = False
        self.board_type = board_type
        self.board_size = board_size
        self.open_cells = open_cells
        self.visualize = False
        self.episode = []
        self.frame_delay = 0

        if board_type == 'diamond':
            self.state = DiamondHexGrid(board_size)
        if board_type == 'triangular':
            self.state = TriangularHexGrid(board_size)

        self._set_empty(self.state, open_cells)
        self.remaining_pegs = self.count_remaining()

    def reset_game(self, visualize=False, frame_delay=0):
        self.finished = False
        self.goal_state = False
        self.visualize = visualize
        self.frame_delay = frame_delay

        if self.board_type == 'diamond':
            self.state = DiamondHexGrid(self.board_size)
        if self.board_type == 'triangular':
            self.state = TriangularHexGrid(self.board_size)

        self._set_empty(self.state, self.open_cells)
        self.remaining_pegs = self.count_remaining()

        if visualize:
            # self.episode.append((copy.deepcopy(self), None))
            self.graph = nx.Graph()
            self.fig = plt.figure(2)
            pos, color_map, node_size = self.state.visualize_grid(self.graph, self, None)
            nx.draw(self.graph, pos, node_color=color_map, node_size=node_size, with_labels=False,
                    font_weight='bold')
            plt.pause(5)


    def visualize_move(self, action):
        self.graph.clear()
        plt.clf()
        pos, color_map, node_size = self.state.visualize_grid(self.graph, self, action)
        nx.draw(self.graph, pos, node_color=color_map, node_size=node_size, with_labels=False,
                font_weight='bold')
        plt.pause(self.frame_delay)

    # TODO: REMOVE
    def visualize_game(self):
        if self.visualize:
            graph = nx.Graph()
            fig2 = plt.figure(2)
            state, action = self.episode[0]
            pos, color_map, node_size = self.state.visualize_grid(graph, state, action)
            nx.draw(graph, pos, node_color=color_map, node_size=node_size, with_labels=False,
                    font_weight='bold')
            plt.pause(5)

            for (state, action) in self.episode[1:]:
                graph.clear()
                plt.clf()
                pos, color_map, node_size = self.state.visualize_grid(graph, state, action)
                nx.draw(graph, pos, node_color=color_map, node_size=node_size, with_labels=False,
                        font_weight='bold')
                plt.pause(self.frame_delay)

            state, _ = self.episode[-1]
            pos, color_map, node_size = self.state.visualize_grid(graph, state, None)
            nx.draw(graph, pos, node_color=color_map, node_size=node_size, with_labels=False,
                    font_weight='bold')
            plt.pause(5)
            fig2.show()


    def flatten_state(self):
        return self.state.flatten()

    def get_legal_actions(self):
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
        action.jump_node.filled = False
        action.start_node.filled = False
        action.end_node.filled = True

        self.remaining_pegs -= 1
        legal_actions = len(self.get_legal_actions())
        if legal_actions == 0:
            self.finished = True

        if self.visualize:
            '''
            self.graph.clear()
            pos, color_map, node_size = self.state.visualize_grid(self.graph, action=action)
            nx.draw(self.graph, pos, node_color=color_map, node_size=node_size, with_labels=False, font_weight='bold')
            plt.pause(2)
            '''
            # self.episode.append((copy.deepcopy(self), copy.deepcopy(action)))
            self.visualize_move(action)

        if legal_actions == 0 and self.remaining_pegs == 1:
            print("congrats, 50 points")
            self.goal_state = True
            self.finished = True
            if self.visualize:
                self.visualize_move(None)
            return 50
        if legal_actions == 0 and self.remaining_pegs > 1:
            self.finished = True
            if self.visualize:
                self.visualize_move(None)
            return -50
        return 0

    # TODO: Dårlig kjøretid
    def is_finished(self):
        return len(self.get_legal_actions()) == 0

    def is_goal_state(self):
        return self.goal_state

    def count_remaining(self):
        return self.state.count_filled()

    @staticmethod
    def _set_empty(state, open_cells):
        for (i, j) in open_cells:
            state.grid[i, j].filled = False


class Action:

    def __init__(self, start_node, jump_node, end_node, name):
        self.start_node = start_node
        self.jump_node = jump_node
        self.end_node = end_node
        self.name = name

    def __str__(self):
        return "{start_node} -> {end_node}".format(start_node=self.start_node.__str__(),
                                                   end_node=self.end_node.__str__())


if __name__ == "__main__":
    s = SimWorld([(1, 1)], board_type="triangular")
    a = s.get_legal_actions()
    for ac in a:
        print(ac)
    print("----------------------------------")

    s.perform_action(a[0])

    a = s.get_legal_actions()
    for ac in a: print(ac)

    print(s.count_remaining())
