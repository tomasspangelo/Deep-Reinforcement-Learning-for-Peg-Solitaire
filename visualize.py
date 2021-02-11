import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from SimWorld import SimWorld

s = SimWorld([(1,1)], board_type="diamond")

G = nx.Graph()

pos = {}
total_height = 1 + 2 * (s.board_size)
total_width = total_height

h = total_height
w = np.ceil(total_width/2)



color_map = []
node_size  = []
for i in range(s.board_size):
    h = total_height - i
    w = np.ceil(total_width/2) - i
    for j in range(s.board_size):
        node = s.state.grid[i, j]
        if node:
            G.add_node(node)
            pos[node] = (w, h)
            if node.filled:
                color_map.append('blue')
                node_size.append(200)
            else:
                color_map.append('grey')
                node_size.append(200)

            h -= 1
            w += 1

for i in range(s.board_size):
    for j in range(s.board_size):
        node = s.state.grid[i, j]
        if node:
            for action in node.neighborhood:
                neighbor = node.neighborhood[action]
                if neighbor:
                    G.add_edge(node, neighbor)





plt.plot()
#pos = nx.spring_layout(G, iterations=100)

nx.draw(G, pos, node_color=color_map, node_size=node_size, with_labels=False, font_weight='bold')


plt.show()
