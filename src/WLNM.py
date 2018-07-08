import primesieve
import numpy as np
import  networkx as nx

class PaletteWL():
    """
    Implementation of the Weisfeiler-Lehman palette coloring of graph
    from Muhan Zhang & Yixin Chen paper KDD'17. It encodes a graph as
    sequences of links enclosing subgraph ordered by graph topology.
    """
    def __init__(self, g, k=10):
        """
        :param g: Networkx graph
        :param k: number of vertices cutoff
        """
        self.graph = g
        self.k = k

    def sub_graph(self, edge):
        """
        Get the subgraph of nodes around the provided initial edge.
        :param edge: tuple(u,v)
        :return: networkx graph
        """
        # select initial nodes
        x, y = edge
        # list of k nodes
        Vk = [x,y]
        # get the neighbors
        while len(Vk) < self.k:
            x_ngb = list(nx.neighbors(self.graph, x))
            y_ngb = list(nx.neighbors(self.graph, y))
            Vk.extend(x_ngb)
            Vk.extend(y_ngb)

        if len(Vk) > self.k:
            Vk_cropped = Vk[:self.k]
            sg = nx.Graph(nx.subgraph(self.graph, Vk_cropped))
        else:
            sg = nx.Graph(nx.subgraph(self.graph, Vk))
        return sg

    def subgraph_colors(self, edge):
        """
        Get the subgraph of nodes around the provided initial edge.
        :param edge: tuple(u,v)
        :return: networkx graph
        """
        # select initial nodes
        x, y = edge
        # list of k nodes
        Vk = [x,y]
        # colors
        colors = [self.dist(x, edge), self.dist(y, edge)]
        # list counter
        ngb_idx = 0
        # get the neighbors
        while len(Vk) < self.k:
            nbg = list(nx.neighbors(self.graph, Vk[ngb_idx]))
            # store the distance of the node from the center as the
            # initial color of the node
            colors.extend([self.dist(i, edge, self.graph) for i in nbg])


    def dist(self):
        pass

    def hashWL(self):
        pass

    def real_to_colors(self):
        pass

    def WLgraphLab(self):
        pass

    def adj_mat_subgraph(self):
        pass

