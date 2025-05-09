# coding : utf-8
# create by ztypl on 2018/5/18


import numpy as np
from gensim.models import Word2Vec


class Graph:
    def __init__(self, filename=None, **kwargs):
        self.node_list = None
        self.head = None
        self.adjTable = None
        if filename:
            self.load_from_numpy(filename)
        self.p = kwargs['p'] if 'p' in kwargs else 1.0
        self.q = kwargs['q'] if 'q' in kwargs else 1.0

    def load_from_numpy(self, file):
        # type: (str) -> None

        arr = np.load(file)
        self.load_from_arr(arr)

    def load_from_arr(self, arr):
        # type: (np.array) -> None

        self.node_list = np.unique(arr)
        self.head = {}
        self.adjTable = []
        for node in self.node_list:
            self.head[node] = -1
        for a, b in arr:
            self._add_edge(a, b)

    def _add_edge(self, a, b):
        # type: (int, int) -> None

        self.adjTable.append((a, b, self.head[a]))
        self.head[a] = len(self.adjTable) - 1

    def have_edge(self, u, v):
        # type: (int, int) -> bool

        next_v_loc = self.head[u]
        while next_v_loc >= 0:
            edge = self.adjTable[next_v_loc]
            if (v == edge[1]):
                return True
            next_v_loc = edge[2]
        return False


    def get_adjacent_vertexs(self, u):
        # type: (int) -> list[int]

        adj_v = []
        next_v_loc = self.head[u]
        while next_v_loc >= 0:
            edge = self.adjTable[next_v_loc]
            adj_v.append(edge[1])
            next_v_loc = edge[2]
        return adj_v

    def node2vec_walk(self, u, t):
        # type: (int, int) -> list[str]

        path = [str(u)]
        for i in range(t - 1):
            cur_node = int(path[-1])
            adj_vertexes = self.get_adjacent_vertexs(cur_node)
            if len(adj_vertexes) == 0:
                break
            if len(path) == 1:
                path.append(str(np.random.choice(adj_vertexes)))
            else:
                prev_node = int(path[-2])
                w = np.zeros(len(adj_vertexes), dtype='float')
                for i, v in enumerate(adj_vertexes):
                    if v == prev_node:
                        w[i] = 1./self.p
                    elif self.have_edge(prev_node, v):
                        w[i] = 1.
                    else:
                        w[i] = 1./self.q
                w = w / w.sum()
                next_node = np.random.choice(adj_vertexes, p=w)
                path.append(str(next_node))
        return path

    def walk(self, w, d, gamma, t, workers):
        # type: (int, int, int, int, int) -> Word2Vec

        walks = []
        v_set = self.node_list.copy()
        for i in range(gamma):
            np.random.shuffle(v_set)
            for v in v_set:
                walks.append(self.node2vec_walk(v, t))

        model = Word2Vec(walks, size=d, window=w, min_count=0, hs=1, sg=1, workers=workers)
        return model
