import primesieve
import numpy as np
import networkx as nx
from random import choice, shuffle

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

    def subgraph_colors(self, g, edge):
        """
        Get the colors of nodes around the provided initial edge.
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
            nbg = list(nx.neighbors(g, Vk[ngb_idx]))
            # store the distance of the node from the center as the
            # initial color of the node
            colors.extend([self.dist(i, edge) for i in nbg])
            # store the nodes
            Vk.extend(nbg)
            # next index
        # crop the lists
        return Vk[:self.k], colors[:self.k]

    def dist(self, v, e):
        """
        Compute the distance with the shortest path length from the edge
        :return:
        """
        sp_x = nx.shortest_path_length(self.graph, v, e[0])
        sp_y = nx.shortest_path_length((self.graph, v, e[1]))
        return np.sqrt(sp_x * sp_y)

    def hashWL(self, x, Vk, colors):
        """
        Weisfeiler & Lehman hashing
        :return:
        """
        position = Vk.index(x)
        # Sum Vk
        s_Vk = np.ceil(np.sum([np.log(primesieve.nth_prime(i)) for i in colors]))
        # sum neighbors colors
        nbg_x = list(nx.neighbors(self.graph, Vk[position]))
        s_nbg_x = np.sum([np.log(primesieve.nth_prime(i)) for i in
                          [colors[Vk.index(y)]]
                          for y in nbg_x if y in Vk])
        return colors[position] + s_nbg_x / s_Vk

    def real_to_colors(self, colors):
        """
        Reduce hash real numbers to sequence of integer
        :param colors:
        :return:
        """
        new_colors = colors.copy()
        # number of colors
        n_cols = len(np.unique(new_colors))
        color_list = []
        for i in range(1, n_cols+1):
            # minimum real value from the initial list
            min_col = np.min(new_colors)
            # consume the hash value
            new_colors.pop(new_colors.index(min_col))
            color_list.append(i)
            # if multiple items have the same hash value
            # add as many item of the same color to the list
            while min_col in new_colors:
                min_col = new_colors.pop(new_colors.index(min_col))
                color_list.append(i)
        return color_list

    def WLgraphLab(self, e):
        """
        Get the subgraph and it's palette-WL colors for ordering
        :return:
        """
        g = self.sub_graph(e)
        Vk, colors = self.subgraph_colors(g, e)
        colors = self.real_to_colors(colors)
        i = 0
        while True:
            new_colors = [self.hashWL(v, Vk, colors) for v in Vk]
            new_colors = self.real_to_colors(new_colors)
            i += 1
            if (new_colors == colors) or i>100:
                break
            else:
                colors = new_colors
        # order Vk by colors
        sorted_Vk = [x for x,_ in sorted(zip(Vk,colors))]
        return sorted_Vk

    def adj_mat_subgraph(self, Vk):
        """"
        Flattened upper triangular adjacency matrix
        """
        a = nx.to_numpy_array(self.graph)
        # select rows
        a = a[Vk,:]
        # keep subgraph nodes columns
        a = a[:, sorted(Vk)]
        # upper triangular matrix without link to predict (a[0,1])
        a = np.triu(a)[:,2:]
        return a.reshape(-1)

    def dataset(self, test_ratio=0.2, val_ratio=0):
        data = []
        edges = list(self.graph.edges())
        negative_edges = []
        for _ in edges:
            while True:
                ne = (choice(list(self.graph.nodes())), choice(list(self.graph.nodes())))
                if ne not in edges:
                    negative_edges.append(ne)
                    break
        for e in edges:
            Vk = self.WLgraphLab(e)
            data.append((1, self.adj_mat_subgraph(Vk)))
        for ne in negative_edges:
            Vk = self.WLgraphLab(ne)
            data.append((0, self.adj_mat_subgraph(Vk)))

        shuffle(data)

        n = len(data)
        train_idx = n - int(n*test_ratio) - int(n*val_ratio)
        test_idx = train_idx + int(n*test_ratio)

        return data[:train_idx,:], data[train_idx:test_idx,:], data[test_idx:,:]

