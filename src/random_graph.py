import networkx as nx
from random import randint,choice, sample
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
from binascii import hexlify
from os import urandom
from math import pi
import math

class RandGraph:
    def __init__(self,
                 actors=5,
                 moving=2,
                 n_entry_nodes=5,
                 n_exit_nodes=4,
                 n_core_nodes=11,
                 n_paths=5,
                 path_depth=6,
                 graph_type=None):

        self.n_paths = n_paths
        self.path_depth = path_depth


        if graph_type == 'simple':
            self.entry_nodes = [1,2]
            self.exit_nodes = [7]
            self.core_nodes = [3,4,5,6]
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(self.entry_nodes)
            self.graph.add_nodes_from(self.exit_nodes)
            capa = [7,6,2,2]
            self.graph.add_nodes_from([(x, {'capacity': capa[i]}) for i,x in enumerate(self.core_nodes)])
            nx.set_node_attributes(self.graph, None, 'actors')
            edges = [
                (1, 3),
                (2, 4),
                (3, 5),
                (4, 5),
                (5, 6),
                (6, 7)
            ]
            self.graph.add_edges_from(edges)

        elif graph_type == 'hongo_district':
            self.entry_nodes = list(range(1, 11))
            self.exit_nodes = list(range(11, 24))
            # core node capacity
            capa = pd.read_csv('../data/core_node_capacity.csv')
            self.core_nodes = capa.node.tolist()
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(self.entry_nodes)
            self.graph.add_nodes_from(self.exit_nodes)
            self.graph.add_nodes_from([(x[1], {'capacity': x[2]}) for x in capa.itertuples()])
            nx.set_node_attributes(self.graph, None, 'actors')

            # entry edges
            entry_edges = pd.read_csv('../data/entry_edges_Hongo_district.csv', header=None)
            entry_edges_tpl = [(x[1], x[2]) for x in entry_edges.itertuples()]
            # exit edges
            exit_edges = pd.read_csv('../data/exit_edges_Hongo_district.csv', header=None)
            exit_edges_tpl = [(x[1], x[2]) for x in exit_edges.itertuples()]
            # core nodes edges
            core_edges = pd.read_csv('../data/core_edge_list_Hongo_district.csv')
            edg_lst = []
            for col in core_edges:
                if col != 'head_node':
                    head_val = core_edges['head_node'].values
                    tail_val = core_edges[col].values
                    edg_lst.append([(x, y) for x, y in zip(head_val, tail_val)])
            core_edge_list = [(x[0], int(x[1])) for y in edg_lst for x in y if not math.isnan(float(x[1]))]
            # add edges
            self.graph.add_edges_from(entry_edges_tpl)
            self.graph.add_edges_from(exit_edges_tpl)
            self.graph.add_edges_from(core_edge_list)

        else:
            self.entry_nodes = list(range(n_entry_nodes))
            n = self.entry_nodes[-1] + 1
            self.exit_nodes = list(range(n, n + n_exit_nodes))
            n = self.exit_nodes[-1] + 1
            self.core_nodes = list(range(n, n + n_core_nodes))
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(self.entry_nodes)
            self.graph.add_nodes_from(self.exit_nodes)
            self.graph.add_nodes_from([(x, {'capacity': randint(1, 10)}) for x in self.core_nodes])
            nx.set_node_attributes(self.graph, None, 'actors')
            self._rand_edges()

        self.actors = self.set_actors(actors)
        self.moving_actors = {}
        self.nb_moving_act = moving
        self.actor_position = {}
        self.step_counter = 0
        self.current_reward = 0

    def actors_position(self, pos):
        h = nx.Graph()
        actor_pos = {}
        for node in self.core_nodes:
            if self.graph.nodes()[node]['actors']:
                nb_actors = len(self.graph.nodes()[node]['actors'])
                for actor in range(nb_actors):
                    name = '%d_%d' % (node, actor)
                    h.add_node(name)
                    x = pos[node][0]
                    x_actor = 0.01 * np.random.randn() + x
                    y = pos[node][1]
                    y_actor = 0.01 * np.random.randn() + y
                    actor_pos[name] = np.array([x_actor, y_actor])
        return h, actor_pos

    def get_node_sizes(self):
        node_sizes = []
        for x, attr in self.graph.nodes(data=True):
            if 'capacity' in attr:
                node_sizes.append(150 + 50 * attr['capacity'])
            else:
                node_sizes.append(150)
        return node_sizes

    def get_node_colors(self):
        node_colors = []
        for x, attr in self.graph.nodes(data=True):
            if x in self.core_nodes:

                if attr['actors'] and (len(attr['actors']) == attr['capacity']):
                    node_colors.append('purple')
                else:
                    node_colors.append('steelblue')
            elif x in self.entry_nodes:
                node_colors.append('g')
            elif x in self.exit_nodes:
                node_colors.append('r')
            else:
                node_colors.append('black')
        return node_colors

    def plot(self, legend=True, fig_size=None):
        if fig_size:
            plt.figure(figsize=fig_size)
        pos = nx.kamada_kawai_layout(self.graph)
        ns = self.get_node_sizes()
        nc = self.get_node_colors()
        nx.draw_networkx_nodes(self.graph,
                               pos,
                               node_color=nc,
                               node_size=ns)
        h, actor_pos = self.actors_position(pos)
        nx.draw_networkx_nodes(h, actor_pos, node_size=20, node_color='orange', alpha=0.7)

        nx.draw_networkx_labels(self.graph, pos, font_color='w')
        nx.draw_networkx_edges(self.graph, pos)
        plt.axis('off')
        #legend
        blue_patch = mpatches.Patch(color='steelblue', label='Core nodes')
        purple_patch = mpatches.Patch(color='purple', label='Saturated nodes')
        red_patch = mpatches.Patch(color='r', label='Exit nodes')
        green_patch = mpatches.Patch(color='g', label='Start nodes')
        orange_patch = mpatches.Patch(color='orange', label='Moving actors')
        if legend:
            plt.legend(handles=[green_patch,
                                blue_patch,
                                purple_patch,
                                red_patch,
                                orange_patch])
        # plt.show()

    def _rand_edges(self):
        for _ in range(self.n_paths):
            edge_list = []

            # select random entry
            entry = choice(self.entry_nodes)

            # random path length
            path_len = randint(2, self.path_depth)

            node1 = entry
            for _ in range(path_len):
                node2 = choice(self.core_nodes)
                edge_list.append((node1, node2))
                node1 = node2

            # add paths to the graph
            self.graph.add_edges_from(edge_list)

            # connect the last node to an exit
            exit_node = choice(self.exit_nodes)
            self.graph.add_edge(node1, exit_node)

    def update_node(self, node, actor):
        act_list = self.graph.nodes[node]['actors']
        # print(node, act_list)
        if not act_list:
            act_list = []
        act_list.append(actor)
        nx.set_node_attributes(self.graph, {node: {'actors': act_list}})


    def remove_from_node(self, node, actor):
        act_list = self.graph.nodes[node]['actors']
        if not act_list:
            act_list = []
        act_list.remove(actor)
        nx.set_node_attributes(self.graph, {node: {'actors': act_list}})

    def set_actors(self, n):
        d = {}
        for i in range(n):
            a = Actor(self.graph, self.entry_nodes, self.exit_nodes)
            d[a.id] = a
        return d

    def get_nb_moving_actors(self, step_nb):
        '''
        Moving actor seasonality according to the step.
        :param step_nb:
        :return: number of moving actors
        '''
        y = int((step_nb + pi) % (8 * pi))
        y /= np.std([0, self.nb_moving_act])*4
        y *= self.nb_moving_act
        y = np.random.normal(loc=y, scale=self.nb_moving_act/10, size=1)
        y = np.round(np.maximum(y,0))
        return int(y[0])

    def init_actors(self, step_nb):
        actor_copy = self.actors

        # get the randomized number of moving actors
        nb_m_a = self.get_nb_moving_actors(step_nb)
        if len(self.actors) >= nb_m_a:
            spl = list(sample(self.actors.keys(), nb_m_a))
        else:
            spl = list(self.actors.keys())

        for n in spl:
            actor = actor_copy.pop(n)
            k = actor.id
            self.moving_actors[k] = actor
            self.moving_actors[k].set_path()
            node = self.moving_actors[k].get_position()
            if node:
                self.update_node(node, k)
            else:
                self.moving_actors.pop(k)
        self.actors = actor_copy

        if (node in self.core_nodes) and (self.graph.nodes[node]['actors']):
            stack = len(self.graph.nodes[node]['actors'])
            return stack < self.graph.nodes[node]['capacity']
        else:
            return True

    def move_actors(self, blocked_nodes):
        actors = self.moving_actors.copy()
        if blocked_nodes:
            bn = blocked_nodes
        else:
            bn = []

        for actor in actors:
            # find current node
            if actor in self.actor_position:
                prev_node = self.actor_position[actor]
            else:
                prev_node = None

            # check if next node is full
            possible_node = self.moving_actors[actor].fetch_next()
            # check that next_node is not in the list of blocked nodes
            if possible_node and (possible_node not in bn):
                if self.get_node_capa(possible_node):
                    # remove id from current node
                    if prev_node:
                        self.remove_from_node(prev_node, actor)
                    # find next node from the path
                    self.moving_actors[actor].move()
                    next_node = self.moving_actors[actor].get_position()
                    # record position
                    self.actor_position[actor] = next_node
                    # add actor to the next node
                    if next_node:
                        self.update_node(next_node, actor)
                    else:
                        self.moving_actors.pop(actor)

    def get_loading(self):
        '''
        Core nodes congestion values
        '''
        values = []
        for node in self.graph.nodes(data=True):
            if 'capacity' in node[1]:
                if node[1]['actors']:
                    values.append(len(node[1]['actors'])/float(node[1]['capacity']))
                else:
                    values.append(0.0)
        return np.array([values])

    def action(self, blocked_nodes=None):
        '''
        Block some nodes and get the next state and reward.
        :param blocked_nodes:
        :return: next_state and reward
        '''
        self.init_actors(self.step_counter)
        self.step_counter += 1
        self.move_actors(blocked_nodes)
        values = self.get_loading()
        reward = self.get_reward()
        return values, reward

    def get_reward(self):
        """
        Reward as a combination of output and congestion level
        :return:
        """
        # number of actors out
        out = sum([len(self.graph.node[n]['actors']) for n
                   in self.exit_nodes
                   if self.graph.node[n]['actors']])
        # saturation of the core nodes
        values = self.get_loading()
        congestion = 1 - np.mean(values)
        # reward combine the evolution of the output and congestion level
        reward = (out - self.current_reward) * congestion
        self.current_reward = out
        return reward

    def step(self, n=10):
        data = np.zeros((1, len(self.core_nodes)))
        for i in range(n):
            curr_prog = round(i/float(n),1)/0.1
            self.init_actors(step_nb=i)
            self.move_actors([])
            values = self.get_loading()
            data = np.vstack((data, values))

            print('\r%% %d %10s' % (round(i/float(n)*100), '|'*int(curr_prog)), sep='',end='', flush=True)
        return data, self.core_nodes

class Actor:
    def __init__(self, g, entry_nodes, exit_nodes):
        self.id = hexlify(urandom(4))
        self.path = None
        self.graph = g
        self.entry_nodes = entry_nodes
        self.exit_nodes = exit_nodes

    def move(self):
        # pop one node from self.path
        if self.path:
            self.path.pop(0)
        else:
            raise ValueError

    def set_path(self):
        # choose shortest path to exit node
        result = None
        while result is None:
            try:
                self.path = nx.shortest_path(self.graph,
                                             source=choice(self.entry_nodes),
                                             target=choice(self.exit_nodes))
                result = True
            except nx.NetworkXNoPath:
                pass

    def get_position(self):
        if self.path:
            return self.path[0]
        else:
            return None
    def fetch_next(self):
        if len(self.path)>1:
            return self.path[1]
        else:
            return None

