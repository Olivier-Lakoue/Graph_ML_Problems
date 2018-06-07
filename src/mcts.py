import numpy as np
import random
import networkx as nx
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import pylab

class GridEnv():
    def __init__(self):
        self.done = False
        self.start = (0,0)
        self.goal = (6,6)
        self.win = False
        self.position = self.start
        self.grid = np.array([
                    [1, 1, 0, 1, 1, 1, 1],
                    [0, 1, 0, 1, 1, 0, 1],
                    [1, 1, 1, 0, 1, 0, 1],
                    [1, 0, 1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 0, 0, 1],
                    [0, 0, 0, 1, 0, 1, 1],
                    [1, 1, 1, 1, 0, 1, 1]
                ])
        self.moves = np.array([[-1, 0],
                               [1, 0],
                               [0, -1],
                               [0, 1]])

    def observe(self):
        idx = np.where((self.moves + self.position).min(axis=1) >= 0)[0]
        return [(i, j) for i, j in self.moves[idx, :] + self.position
                if np.subtract(self.grid.shape, (i, j)).all()
                and self.grid[i, j] >= 0]

    def action(self, next_position):
        if self.grid[next_position] == 0:
            self.done = True
        elif next_position == self.goal:
            self.win = True
            self.done = True
        else:
            self.position = next_position

class MCTS():
    def __init__(self):
        self.g = nx.DiGraph()
        self.C = np.sqrt(2)
        self.env = GridEnv()
        self.path = []
        self.g.add_node(self.env.start, wins=0, plays=0)

    def expand(self, current, positions):
        self.g.add_nodes_from(positions, wins=0, plays=0)
        self.g.add_edges_from([(current, pos) for pos in positions])

    def simulate(self, max_moves=1000, strategy='uct', step=100):

        for t in range(max_moves):
            current = self.env.position
            possible_positions = self.env.observe()
            if self.env.done:
                self.env = GridEnv()
                self.path = []

            # choose next move
            # random if not known
            if not (np.array([(p in self.g) for p in possible_positions])).all():
                self.expand(current, possible_positions)
                position = random.choice(possible_positions)
            elif strategy == 'uct':
            # or using the Upper Confidence bound applied to Trees (uct) strategy
                log_nb_sims = np.log(t+1)
                nb_plays = np.array([self.g.nodes(data=True)[pos]['plays']
                                     for pos in possible_positions]) + 1
                nb_wins = np.array([self.g.nodes(data=True)[pos]['wins']
                                    for pos in possible_positions]) + 1
                wi_ni = np.divide(nb_wins, nb_plays)
                ln_ni = self.C * np.sqrt(np.divide(log_nb_sims, nb_plays))
                utc = np.add(wi_ni, ln_ni)
                position = possible_positions[np.argmax(utc)]
            else:
                position = random.choice(possible_positions)
            # save path
            self.path.append(position)

            # update count
            self.g.nodes(data=True)[position]['plays'] += 1

            # move in the env
            self.env.action(position)

            # update wins
            if self.env.win:
                for node in self.path:
                    self.g.nodes(data=True)[node]['wins'] += 1

            if t % step == 0:
                self.plot(strategy)
            if t == (max_moves - 1):
                plt.pause(5)

    def plot(self, strategy):
        pylab.clf()
        if strategy == 'uct':
            plt.title('Monte Carlo Tree Search Strategy')
        else:
            plt.title(' %s Search Strategy' % strategy)

        pos = {node: np.array([node[1], node[0]]) for node in self.g.nodes()}
        sizes = [attr['plays'] for node, attr in self.g.nodes(data=True)]
        wins = [attr['wins'] for node, attr in self.g.nodes(data=True)]
        norm = Normalize(vmin=min(wins)-1, vmax=max(wins), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat_r)
        cols = [mapper.to_rgba(v) for v in wins]

        nx.draw_networkx_nodes(self.g, pos,
                               node_size=sizes,
                               alpha=0.7,
                               linewidths=0,
                               node_color=cols)

        plt.imshow(self.env.grid, cmap='gray')
        plt.yticks([])
        plt.xticks([])
        plt.axis('off')

        start_pos = self.env.start
        goal_pos = self.env.goal
        for node in self.g.nodes():
            if node == start_pos:
                start_pos = (node[1], node[0])
            if node == goal_pos:
                goal_pos = (node[1], node[0])

        plt.text(*start_pos, 'Start')
        plt.text(*goal_pos, 'Goal')
        pylab.draw()
        plt.pause(0.1)

uct = MCTS()
rand = MCTS()

rand.simulate(max_moves=10000, strategy='Random')
uct.simulate(max_moves=10000)