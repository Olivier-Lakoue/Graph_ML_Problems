import sys
sys.path.append('../src/')
from random_graph import RandGraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
plt.style.use('seaborn-talk')

g = RandGraph(graph_type='medium', actors=1000)
g_congest = []
h = RandGraph(graph_type='medium', actors=1000)
h_congest = []
steps = 200

for i in range(steps):
    pylab.clf()

    plt.subplot(221)
    plt.title('No action')

    g_cong,reward = g.action(0,0.0)
    g_congest.append(np.mean(g_cong))
    g.plot(legend=False)

    plt.subplot(222)
    plt.title('Random action')
    act = np.random.choice(range(len(g.controllable_intersections)))
    val = np.random.randn()
    h_cong, reward = h.action(act, val)
    h_congest.append(np.mean(h_cong))
    h.plot(legend=False)

    plt.subplot(223)
    plt.title('Congestion level')
    df_g = pd.DataFrame(g_congest)
    plt.plot(df_g)
    plt.xlim((0,steps))
    plt.ylim((0,1))

    plt.subplot(224)
    plt.title('Congestion level')
    df_h = pd.DataFrame(h_congest)
    plt.plot(df_h)
    plt.xlim((0, steps))
    plt.ylim((0, 1))

    pylab.draw()
    plt.pause(0.2)