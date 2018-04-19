import networkx as nx
from random import randint,choice
import matplotlib.pyplot as plt


class RandGraph:
    def __init__(self, n_entry_nodes=5, n_exit_nodes=4, n_core_nodes=11, n_paths=5, path_depth=6):
        self.n_paths = n_paths
        self.path_depth = path_depth
        self.entry_nodes = list(range(n_entry_nodes))
        n = self.entry_nodes[-1]
        self.exit_nodes = list(range(n, n + n_exit_nodes))
        n = self.exit_nodes[-1]
        self.core_nodes = list(range(n, n + n_core_nodes))
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.entry_nodes)
        self.graph.add_nodes_from(self.exit_nodes)
        self.graph.add_nodes_from([(x, {'capacity': randint(1, 10)}) for x in self.core_nodes])
        self._rand_edges()

    def plot(self):
        pos = nx.kamada_kawai_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_color='steelblue')
        nx.draw_networkx_labels(self.graph, pos, font_color='w')
        nx.draw_networkx_edges(self.graph, pos)
        plt.axis('off')
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
                edge_list.append((node1,node2))
                node1 = node2

            # add paths to the graph
            self.graph.add_edges_from(edge_list)

            # connect the last node to an exit
            exit_node = choice(self.exit_nodes)
            self.graph.add_edge(node1, exit_node)


class Actor:
    def __init__(self):
        self.id = randint(0, 1000)
        self.path = None

    def move(self):
        # pop one node from self.path
        if self.path:
            self.path.pop(0)
        else:
            raise ValueError

    def set_path(self, g):
        # choose shortest path to exit node
        while not self.path:
            try:
                self.path = nx.shortest_path(g.graph,
                                             source=choice(g.entry_nodes),
                                             target=choice(g.exit_nodes))
            except nx.NetworkXNoPath:
                pass

    def get_position(self):
        if self.path:
            return self.path[0]
        else:
            raise ValueError


class BunchActors:
    def __init__(self, n, g):
        self.nb_actors = n
        self.bunch = {}
        self._get_bunch()
        self._set_paths(g)

    def _get_bunch(self):
        for _ in range(self.nb_actors):
            a = Actor()
            self.bunch[a.id] = a

    def ids(self):
        return list(self.bunch.keys())

    def get_actor(self, n):
        if n in self.ids():
            return self.bunch[n]
        else:
            return None

    def _set_paths(self, g):
        for actor in self.bunch.values():
            actor.set_path(g)

    def get_positions(self):
        d = {}
        for idx, actor in self.bunch.items():
            try:
                d[idx] = actor.get_position()
            except ValueError:
                pass
        return d

    def move(self):
        to_remove = {}
        for idx, actor in self.bunch.items():
            try:
                actor.move()
            except ValueError:
                to_remove[idx] = actor
        self.bunch = {k: v for k, v in self.bunch.items() if k not in to_remove}
        self.nb_actors -= len(to_remove)