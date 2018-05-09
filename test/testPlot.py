import sys
import random
sys.path.append('../src/')
from random_graph import RandGraph
import matplotlib.pyplot as plt
import pylab

g = RandGraph(graph_type='simple', actors=100)
for i in range(20):
    pylab.clf()
    g.action()
    g.plot()
    pylab.draw()
    plt.pause(0.5)
