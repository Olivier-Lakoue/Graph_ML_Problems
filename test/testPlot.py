import sys
import random
sys.path.append('../src/')
from random_graph import RandGraph
from deep_q_learning import DQN
import matplotlib.pyplot as plt
import pylab
import pandas as pd

g = RandGraph(graph_type='simple', actors=1000)
h = RandGraph(graph_type='simple', actors=1000)
dqn_g = DQN(g)
dqn_h = DQN(h)
dqn_h.load()

plt.style.use('seaborn-talk')


for i in range(30):

    pylab.clf()

    plt.subplot(121)
    plt.title('Random action')
    rand_act = dqn_g.action_space.sample()
    a1 = dqn_g.action_space.get_nodes(rand_act)
    g.action(a1)
    g.plot()


    plt.subplot(122)
    plt.title('Deep Q Learning')
    state = h.get_loading()
    a2 = dqn_h.predict(state)
    h.action(a2)
    h.plot(legend=False)

    pylab.draw()
    plt.pause(0.3)
