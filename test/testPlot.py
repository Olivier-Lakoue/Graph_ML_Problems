import sys
import random
sys.path.append('../src/')
from random_graph import RandGraph

g = RandGraph(graph_type='simple')
g.plot()