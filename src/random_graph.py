import networkx as nx
from random import randint,choice
import matplotlib.pyplot as plt





class RandGraph:
    def __init__(self,actors=5, moving=2, n_entry_nodes=5, n_exit_nodes=4, n_core_nodes=11, n_paths=5, path_depth=6):
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
        nx.set_node_attributes(self.graph, None, 'actors')
        self._rand_edges()
        self.actors = self.set_actors(actors)
        self.moving_actors = {}
        self.nb_moving_act = moving

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

    def update_actor_list(self, key, node, prev_node=None):
        if prev_node and self.graph.nodes[prev_node]['actors']:
            self.graph.nodes[prev_node]['actors'].remove(key)
        if node and self.graph.nodes[node]['actors']:
            self.graph.nodes[node]['actors'].append(key)
        elif prev_node and not node and self.graph.nodes[prev_node]['actors']:
            if key in self.graph.nodes[prev_node]['actors']:
                self.graph.nodes[prev_node]['actors'].remove(key)
        elif node:
            self.graph.nodes[node]['actors'] = [key]
        else:
            pass


    def set_actors(self, n):
        d = {}
        for i in range(n):
            a = Actor(self.graph, self.entry_nodes, self.exit_nodes)
            d[a.id] = a
        return d

    def init_actors(self):
        # move actors and update nodes actors list
        keys = [a.id for a in self.actors.values()][:self.nb_moving_act]
        for key in keys:

            self.moving_actors[key] = self.actors.pop(key)
            # set actor path
            self.moving_actors[key].set_path()
            print(self.moving_actors[key].path)
            # record actors into nodes
            node = self.moving_actors[key].get_position()
            # print(key, node)
            self.update_actor_list(key, node)

    def move_actors(self):
        to_remove = []
        # pop current path position for each moving_actors
        for key, actor in self.moving_actors.items():
            prev_node = actor.get_position()
            actor.move()
            node = actor.get_position()
            # update nodes actors lists
            if node:
                self.update_actor_list(key, node, prev_node)
            else:
                to_remove.append(key)
                self.update_actor_list(key, node, prev_node)
        # remove leaving actors
        [self.moving_actors.pop(i) for i in to_remove]


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
        while not self.path:
            try:
                self.path = nx.shortest_path(self.graph,
                                             source=choice(self.entry_nodes),
                                             target=choice(self.exit_nodes))
            except nx.NetworkXNoPath:
                pass

    def get_position(self):
        if self.path:
            return self.path[0]
        else:
            return None


