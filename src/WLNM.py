# import primesieve
import numpy as np
import networkx as nx
from random import choice, shuffle

class PaletteWL():
    """
    Implementation of the Weisfeiler-Lehman palette coloring of graph
    from Muhan Zhang & Yixin Chen paper KDD'17. It encodes a graph as
    sequences of links enclosing subgraph ordered by graph topology.
    """
    def __init__(self, g, k=10, encoding_len=100):
        """
        :param g: Networkx graph
        :param k: number of vertices cutoff
        """
        self.graph = g
        self.k = k
        p = [2,  3,  5,  7,  11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
                       73,  79,  83,  89,  97,  101,  103,  107,  109,  113,  127,  131,  137,  139,  149,  151,  157,
                       163,  167,  173,  179,  181,  191,  193,  197,  199,  211,  223,  227,  229,  233,  239,  241,
                       251,  257,  263,  269,  271,  277,  281,  283,  293,  307,  311,  313,  317,  331,  337,  347,
                       349,  353,  359,  367,  373,  379,  383,  389,  397,  401,  409,  419,  421,  431,  433,  439,
                       443,  449,  457,  461,  463,  467,  479,  487,  491,  499,  503,  509,  521,  523]
        self.primes = {i:n for i,n in enumerate(p)}
        self.maxlen = encoding_len

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
        try:
            sp_x = nx.shortest_path_length(self.graph, v, e[0])
        except nx.NetworkXNoPath:
            # an arbitrary high value
            sp_x = 1000
        try:
            sp_y = nx.shortest_path_length(self.graph, v, e[1])
        except nx.NetworkXNoPath:
            sp_y = 1000
        return np.sqrt(sp_x * sp_y)

    def hashWL(self, x, Vk, colors):
        """
        Weisfeiler & Lehman hashing
        :return:
        """
        position = Vk.index(x)
        # Sum Vk
        s_Vk = np.ceil(np.sum([np.log(self.primes[i]) for i in colors]))
        # sum neighbors colors
        nbg_x = list(nx.neighbors(self.graph, Vk[position]))
        s_nbg_x = np.sum([np.log(self.primes[i]) for i in [colors[Vk.index(y)] for y in nbg_x if y in Vk]])
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

    def adj_mat_subgraph_with_features(self, Vk):
        """
        Flattened upper triangular adjacency matrix with an extra dim for
        nodes encoding features
        :param Vk:
        :return: numpy array of shape (:,self.maxlen)
        """
        a = nx.to_numpy_array(self.graph)
        # select rows
        a = a[Vk,:]
        # keep subgraph nodes columns
        a = a[:, sorted(Vk)]
        # add a dim to align with feature matrix
        a = np.expand_dims(a, axis=2)

        # feature matrix
        feat = np.zeros((0, self.maxlen))
        for n in Vk:
            feat = np.vstack(feat, self.graph.node[n]['encoding'])
        # remove the primer
        feat = feat[1:, :]
        # insert a dim to align with adjacency mat
        feat = np.expand_dims(feat, axis=0)

        # combine matrices
        res = np.multiply(a, feat)
        # remove link to predict
        res = np.triu(res)[:, 2:]
        return res.reshape(-1, res.shape[-1])

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

        train = data[:train_idx]
        test = data[train_idx:test_idx]
        val = data[test_idx:]

        y_train = np.array([y for y,_ in train])
        x_train = np.array([x for _,x in train])
        y_test = np.array([y for y,_ in test])
        x_test = np.array([x for _, x in test])
        y_val = np.array([y for y, _ in val])
        x_val = np.array([x for _, x in val])

        return (y_train,x_train), (y_test, x_test), (y_val, x_val)

