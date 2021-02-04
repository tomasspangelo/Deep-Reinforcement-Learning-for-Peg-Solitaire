import networkx as nx
import matplotlib.pyplot as plt

from SimWorld import SimWorld

s = SimWorld([(1,1)], board_type="diamond")

G = nx.Graph()

for i in range(s.board_size):
    for j in range(s.board_size):
        node = s.state.grid[i, j]
        if node:
            G.add_node(node)

for i in range(s.board_size):
    for j in range(s.board_size):
        node = s.state.grid[i, j]
        if node:
            for action in node.neighborhood:
                neighbor = node.neighborhood[action]
                if neighbor:
                    G.add_edge(node, neighbor)





plt.plot()
pos = nx.spring_layout(G, iterations=100)
nx.draw(G, pos, with_labels=True, font_weight='bold')


plt.show()
