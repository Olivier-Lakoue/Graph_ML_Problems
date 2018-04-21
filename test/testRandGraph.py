from src.random_graph import RandGraph
import networkx as nx

g = RandGraph()

g.init_actors()
print(g.graph.nodes(data=True))
g.move_actors()
print(g.graph.nodes(data=True))
