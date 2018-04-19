from src.random_graph import RandGraph
import networkx as nx

g = RandGraph(n_paths=5, path_depth=6)

# print(g.core_nodes)
# print(g.exit_nodes)
# print(g.entry_nodes)
print(nx.degree(g.graph))

g.plot()
