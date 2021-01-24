from HexGrid import DiamondHexGrid, TriangularHexGrid


class SimWorld:

    def __init__(self, open_cells, board_type='diamond', board_size=4, visualize=False):
        self.finished = False
        self.board_size = board_size
        if board_type == 'diamond':
            self.state = DiamondHexGrid(board_size)
        if board_type == 'triangular':
            self.state = TriangularHexGrid(board_size)

        self._set_empty(self.state, open_cells)
        self.remaining_pegs = self.count_remaining()

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
                            legal_actions.append(Action(start_node=node, jump_node=neighbor, end_node=next_neighbor, name=action))

        return legal_actions

    def perform_action(self, action):
        action.jump_node.filled = False
        action.start_node.filled = False
        action.end_node.filled = True

        self.remaining_pegs -= 1
        legal_actions = len(self.get_legal_actions())
        if legal_actions == 0:
            self.finished = True

        if legal_actions == 0 and remaining_pges == 1:
            return 100
        return 0 if legal_actions > 0 else -100




    def is_finished(self):
        return self.finished

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
        return "{start_node} -> {end_node} ({name})".format(start_node=self.start_node.__str__(),
                                                            end_node=self.end_node.__str__(),
                                                            name=self.name)


if __name__ == "__main__":
    s = SimWorld([(1,1)], board_type="triangular")
    a = s.get_legal_actions()
    for ac in a:
        print(ac)
    print("----------------------------------")

    s.perform_action(a[0])

    a = s.get_legal_actions()
    for ac in a: print(ac)

    print(s.count_remaining())

