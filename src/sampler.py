import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from scipy import sparse
import networkx as nx
import random
from utils import RandomizedFilter

class SparseUniformNeighborSampler(object):
    def __init__(self, graph: sparse.csr_matrix):
        assert sparse.issparse(graph), "SparseUniformNeighborSampler: not sparse.issparse(graph)"
        self.g = graph

    def __pad_sample_node(self, nodes_list: list, sample_size: int):
        if len(nodes_list) > sample_size:
            nodes_list = list(np.random.choice(nodes_list, size=sample_size, replace=False))
        else:
            nodes_list.extend(list(np.random.choice(nodes_list, size=sample_size - len(nodes_list))))
        return nodes_list

    def __call__(self, ids: torch.LongTensor, n_samples=3):
        neighbor = []
        ids = ids.tolist()
        for userid in ids:
            tmp_list = self.g[userid].nonzero()[1].tolist()
            if len(tmp_list) == 0:
                tmp_list.append(userid)
            neighbor.append(self.__pad_sample_node(tmp_list, n_samples))

        neighbor = np.array(neighbor)
        return Variable(torch.LongTensor(neighbor))


class SparseUniformNeighborMetaSampler(object):
    def __init__(self, graph: sparse.csr_matrix, schema, nodetpye):
        assert sparse.issparse(graph), "SparseUniformNeighborSampler: not sparse.issparse(graph)"
        self.g = graph
        self.schema = schema
        self.nodetype = nodetpye

    def __pad_sample_node(self, nodes_list: list, sample_size: int):
        if len(nodes_list) > sample_size:
            nodes_list = list(np.random.choice(nodes_list, size=sample_size, replace=False))
        else:
            nodes_list.extend(list(np.random.choice(nodes_list, size=sample_size - len(nodes_list))))
        return nodes_list

    def __call__(self, ids: torch.LongTensor, index, n_samples=3):
        neighbor = []
        ids = ids.tolist()
        for userid in ids:
            tmp_list = self.g[userid].nonzero()[1].tolist()
            tmp_list = [node for node in tmp_list if self.nodetype[node] == self.schema[index % (len(self.schema) - 1)]]
            if len(tmp_list) == 0:
                tmp_list.append(userid)
            neighbor.append(self.__pad_sample_node(tmp_list, n_samples))

        neighbor = np.array(neighbor)
        return Variable(torch.LongTensor(neighbor))


class UniformNeighborMetaSampler(object):
    def __init__(self, graph, schema, nodetpye):
        self.g = self.build_graph(graph)
        self.schema = schema
        self.nodetype = nodetpye

    def build_graph(self, g):
        graph = nx.Graph()
        for (x, y) in g:
            graph.add_edge(x, y)
        return graph

    def __pad_sample_node(self, nodes_list: list, sample_size: int):
        if len(nodes_list) > sample_size:
            nodes_list = list(np.random.choice(nodes_list, size=sample_size, replace=False))
        else:
            nodes_list.extend(list(np.random.choice(nodes_list, size=sample_size - len(nodes_list))))
        return nodes_list

    def __call__(self, ids: torch.LongTensor, index, n_samples=3):
        neighbor = []
        ids = ids.tolist()
        for userid in ids:
            if userid in self.g:
                tmp_list = [n for n in self.g[userid]]
                if self.schema is not None and self.nodetype is not None:
                    tmp_list = [node for node in tmp_list if self.nodetype[node] == self.schema[index % (len(self.schema))]]
            else:
                tmp_list = []
            if len(tmp_list) == 0:
                tmp_list.append(userid)
            neighbor.append(self.__pad_sample_node(tmp_list, n_samples))

        neighbor = np.array(neighbor)
        return Variable(torch.LongTensor(neighbor))


class UniformRandomSampler(object):
    def __init__(self, graphs, nodetpye, filter_type, filter_list, sample_layer):
        self.gs = graphs
        self.nodetype = nodetpye
        self.filter_type = filter_type
        self.accpeted_filter_list = filter_list
        self.sample_layer = sample_layer

    def build_graph(self, g):
        graph = nx.Graph()
        for (x, y) in g:
            graph.add_edge(x, y)
        return graph

    def __pad_sample_node(self, nodes_list: list, filter_list: list, sample_size: int):
        assert len(nodes_list) == len(filter_list), 'nodes_list and filter_list must be correspond, get {}, {}'\
            .format(len(nodes_list), len(filter_list))
        idx = [i for i in range(len(nodes_list))]
        random.shuffle(idx)
        if len(nodes_list) > sample_size:
            selected_idx = idx[:sample_size]
            nodes_list = [nodes_list[sid] for sid in selected_idx]
            filter_list = [filter_list[sid] for sid in selected_idx]
        else:
            while len(nodes_list) < sample_size:
                rand_val = random.choice(idx)
                nodes_list.append(nodes_list[rand_val])
                filter_list.append(filter_list[rand_val])
        return nodes_list, filter_list

    def __call__(self, ids: torch.LongTensor, index, filter_list, n_samples=3):
        '''
        :param ids:
        :param index:
        :param filter_list: ["U0I0U", "U0I1U"]
        :param n_samples:
        :return:
        '''
        neighbor = []
        total_filter_list = []
        ids = ids.tolist()
        for ids_idx, userid in enumerate(ids):
            tmp_list = []
            cur_filter_list = []
            for typeid, g in enumerate(self.gs):
                if userid in g:
                    g_neighbrs = [n for n in g[userid]]
                    if n_samples != self.sample_layer[-1]:
                        tmp_filter_list = [filter_list[ids_idx] + str(typeid) + self.nodetype[n] for n in g_neighbrs]
                        tmp_list.extend(g_neighbrs)
                        cur_filter_list.extend(tmp_filter_list)
                    else:
                        # judge
                        if self.filter_type == RandomizedFilter.IDENTITY or self.filter_type is None:
                            tmp_filter_list = [filter_list[ids_idx] + str(typeid) + self.nodetype[n] for n in g_neighbrs]
                            tmp_list.extend(g_neighbrs)
                            cur_filter_list.extend(tmp_filter_list)
                        elif self.filter_type == RandomizedFilter.RELATION:
                            func = lambda x: True if filter_list[ids_idx][-2] != str(typeid) else False
                            g_neighbrs = list(filter(func, g_neighbrs))
                            tmp_filter_list = [filter_list[ids_idx] + str(typeid) + self.nodetype[n] for n in g_neighbrs]
                            tmp_list.extend(g_neighbrs)
                            cur_filter_list.extend(tmp_filter_list)
                        elif self.filter_type == RandomizedFilter.METAPATH:
                            func = lambda x: True if filter_list[ids_idx] + str(typeid) + self.nodetype[x] not in self.accpeted_filter_list else False
                            g_neighbrs = list(filter(func, g_neighbrs))
                            tmp_filter_list = [filter_list[ids_idx] + str(typeid) + self.nodetype[n] for n in g_neighbrs]
                            tmp_list.extend(g_neighbrs)
                            cur_filter_list.extend(tmp_filter_list)
                        else:
                            pass
            if len(tmp_list) == 0:
                tmp_list.append(userid)
                cur_filter_list.append(self.nodetype[userid])
            nlist, flist = self.__pad_sample_node(tmp_list, cur_filter_list, n_samples)
            neighbor.append(nlist)
            total_filter_list.extend(flist)

        neighbor = np.array(neighbor)
        return Variable(torch.LongTensor(neighbor)), total_filter_list


if __name__ == '__main__':
    a = sparse.csr_matrix((np.ones(5), (np.array([0, 0, 1, 1, 2]), np.array([0, 1, 2, 3, 4]))), shape=(5, 5))
    s = SparseUniformNeighborSampler(a)
    b = torch.tensor([3, 4, 2]).long()
    d = s(b)
    print(d)
    g = [(1, 2), (1, 3), (1, 4), (1, 5)]
    unisampler = UniformNeighborMetaSampler(graph=g, schema=None, nodetpye=None)
    print([n for n in unisampler.build_graph(g)[1]])
    print(unisampler(torch.tensor([1]).long(), index=0))

