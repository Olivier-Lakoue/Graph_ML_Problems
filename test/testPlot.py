import sys
import random
sys.path.append('../src/')
from random_graph import RandGraph
import matplotlib.pyplot as plt
import pylab

g = RandGraph(graph_type='simple', actors=1000)
h = RandGraph(graph_type='simple', actors=1000)
plt.style.use('seaborn-talk')


for i in range(30):

    pylab.clf()

    plt.subplot(121)
    g.action()
    g.plot()


    plt.subplot(122)
    h.action()
    h.plot()

    pylab.draw()
    plt.pause(0.3)
