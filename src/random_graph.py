import networkx as nx
from random import randint,choice, sample
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

class RandGraph:
    def __init__(self, actors=5, moving=2, n_entry_nodes=5, n_exit_nodes=4, n_core_nodes=11, n_paths=5, path_depth=6):
        self.n_paths = n_paths
        self.path_depth = path_depth
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

    def plot(self):
        pos = nx.kamada_kawai_layout(self.graph)
        nx.draw_networkx_nodes(self.graph,
                               pos,
                               node_color='steelblue',
                               nodelist=self.core_nodes)
        nx.draw_networkx_nodes(self.graph,
                               pos,
                               node_color='g',
                               nodelist=self.entry_nodes)
        nx.draw_networkx_nodes(self.graph,
                               pos,
                               node_color='r',
                               nodelist=self.exit_nodes)

        nx.draw_networkx_labels(self.graph, pos, font_color='w')
        nx.draw_networkx_edges(self.graph, pos)
        plt.axis('off')
        #legend
        blue_patch = mpatches.Patch(color='steelblue', label='Core nodes')
        red_patch = mpatches.Patch(color='r', label='Exit nodes')
        green_patch = mpatches.Patch(color='g', label='Start nodes')

        plt.legend(handles=[green_patch,blue_patch,red_patch])
        plt.show()

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

    def init_actors(self):
        actor_copy = self.actors
        if len(self.actors) >= self.nb_moving_act:
            spl = list(sample(self.actors.keys(), self.nb_moving_act))
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

    def get_node_capa(self, node):
        if (node in self.core_nodes) and (self.graph.nodes[node]['actors']):
            stack = len(self.graph.nodes[node]['actors'])
            return stack < self.graph.nodes[node]['capacity']
        else:
            return True

    def actor_position(self, key):
        for n in self.graph.nodes(data=True):
            if (n[1]['actors']) and key in n[1]['actors']:
                return n[0]

    def move_actors(self):
        actors = self.moving_actors.copy()
        for actor in actors:
            # find current node
            prev_node = self.actor_position(actor)
            #     print(prev_node)
            # check if next node is full
            possible_node = self.moving_actors[actor].fetch_next()
            if possible_node:
                if self.get_node_capa(possible_node):
                    # remove id from current node
                    self.remove_from_node(prev_node, actor)
                    # find next node from the path
                    self.moving_actors[actor].move()
                    next_node = self.moving_actors[actor].get_position()
                    # add actor to the next node
                    if next_node:
                        self.update_node(next_node, actor)
                    else:
                        self.moving_actors.pop(actor)

    def get_loading(self):
        values = []
        for node in self.graph.nodes(data=True):
            if 'capacity' in node[1]:
                if node[1]['actors']:
                    values.append(len(node[1]['actors'])/float(node[1]['capacity']))
                else:
                    values.append(0.0)
        return np.array([values])

    def get_reward(self, type='rm'):
        if (type == 'rm') :
            denom = np.sum(list(nx.get_node_attributes(self.graph, 'capacity').values()), dtype=float)
            num = np.sum([len(x) for x in nx.get_node_attributes(self.graph, 'actors').values() if x])
            ratio = num / denom
            reward = 1.0 - ratio
            return reward
        elif (type == 'mr'):
            denom = np.array(list(nx.get_node_attributes(self.graph, 'capacity').values()), dtype=float)
            num = np.array([len(x) if x else 0 for x in nx.get_node_attributes(self.graph.subgraph(self.core_nodes), 'actors').values()])
            ratio = np.mean(num / denom)
            reward = 1.0 - ratio
            return reward


    def step(self, n=10, reward_type='rm'):
        data = np.zeros((1, len(self.core_nodes)))
        reward = []
        for i in range(n):
            self.init_actors()
            self.move_actors()
            values = self.get_loading()
            data = np.vstack((data, values))
            reward.append(self.get_reward(type=reward_type))
        return data, self.core_nodes, reward

class Actor:
    def __init__(self, g, entry_nodes, exit_nodes):
        self.id = randint(0, 1000)
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

