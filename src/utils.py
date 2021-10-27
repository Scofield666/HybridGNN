import argparse
from collections import defaultdict

import networkx as nx
import numpy as np
from gensim.models.keyedvectors import Vocab
from six import iteritems
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)
import torch

from walk import RWGraph, RWGraphs
from random_walks import RandomWalk

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--regloss', type=bool, default=False)
    parser.add_argument('--att_vis', type=bool, default=False)
    parser.add_argument('--input', type=str, default='../data/amazon_processed',
                        help='Input dataset path')

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch. Default is 100.')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of batch_size. Default is 64.')

    parser.add_argument('--eval_type', type=str, default='all',
                        help='The edge type(s) for evaluation.')
    # U - I - U, I - U - I
    parser.add_argument('--schema', type=str, default='I-I-I',
                        help='The metapath schema (e.g., U-I-U,I-U-I).')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--edge_dim', type=int, default=8,
                        help='Number of edge embedding dimensions. Default is 8.')

    parser.add_argument('--walk_length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num_walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')

    parser.add_argument('--window_size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--negative_samples', type=int, default=5,
                        help='Negative samples for optimization. Default is 5.')

    parser.add_argument('--edge_type_count', type=int, default=2,
                        help='Edge type count. Default is 4.')

    parser.add_argument('--random_walk_length', type=int, default=10,
                        help='Random walk length. Default is 3.')
    parser.add_argument('--random_sample_layers', type=int, default=3,
                        help='Random walk length. Default is 3.')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='Learning rate. Default is 5e-3.')
    parser.add_argument('--percent', type=int, default=0,
                        help='percentage of train size. Default is 0.')
    parser.add_argument('--upto', nargs='+', type=int, default=[-1])
    # parser.add_argument('--upto', type=int, default=-1,
    #                     help='relations that can be seen. -1 means no restriction. Default is -1')
    parser.add_argument('--maintype', type=int, default=-1,
                        help='single relation optimized. -1 means no restriction. Default is -1')
    parser.add_argument('--filter', type=str, default=None,
                        help='filter type of random walk. Default is None (No filtering)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')
    parser.add_argument('--rm_rand', type=bool, default=False,
                        help='remove rand if edge type=1 or all metapaths in predefined')

    return parser.parse_args()

class RandomizedFilter():
    IDENTITY = "identity"
    METAPATH = "metapath"
    RELATION = "relation"
    def __init__(self, metapath, nodetype, filter_type=None):
        self.metapath_list = metapath.split(",")
        self.nodetype = nodetype
        self.filter_type = filter_type

    '''
        candidates: [[[walk], [edges]], [[walk], [edges]], ...]
    '''
    def filter_by_metapath(self, candidates):
        assert len(candidates) > 0
        func = lambda x: True if '-'.join([self.nodetype[xx] for xx in x[0]]) not in self.metapath_list else False
        candidates = filter(func, candidates)
        return list(candidates)

    def filter_by_single_relation(self, candidates):
        assert len(candidates) > 0
        func = lambda x: True if any([x[1][0] != xx for xx in x[1]]) else False
        candidates = filter(func, candidates)
        return list(candidates)

    def filter_identity(self, candidates):
        return candidates

    def filter(self, candidates):
        print("Before filtering: ", len(candidates))
        if self.filter_type == RandomizedFilter.IDENTITY or self.filter_type == None:
            candidates = self.filter_identity(candidates)
            print("After filtering by {} : {} ".format(self.filter_type, len(candidates)))
            return candidates
        elif self.filter_type == RandomizedFilter.METAPATH:
            candidates = self.filter_by_metapath(candidates)
            print("After filtering by {} : {} ".format(self.filter_type, len(candidates)))
            return candidates
        elif self.filter_type == RandomizedFilter.RELATION:
            candidates = self.filter_by_single_relation(candidates)
            print("After filtering by {} : {} ".format(self.filter_type, len(candidates)))
            return candidates
        else:
            pass

    def removeEdgeInfo(self, candidates):
        return [cand[0] for cand in candidates]

def set_seeds(seed=0):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed(seed)


def get_G_from_edges(edges):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.Graph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        x = int(edge_key.split('_')[0])
        y = int(edge_key.split('_')[1])
        tmp_G.add_edge(x, y)
        tmp_G[x][y]['weight'] = weight
    return tmp_G

def get_Gs_from_edges(graph_by_type):
    Gs = []
    edge_dict = dict()
    for typeid, edges in graph_by_type.items():
        for edge in edges:
            edge_key = str(edge[0]) + '_' + str(edge[1])
            if edge_key not in edge_dict:
                edge_dict[edge_key] = 1
            else:
                edge_dict[edge_key] += 1
        tmp_G = nx.Graph()
        for edge_key in edge_dict:
            weight = edge_dict[edge_key]
            x = int(edge_key.split('_')[0])
            y = int(edge_key.split('_')[1])
            tmp_G.add_edge(x, y)
            tmp_G[x][y]['weight'] = weight
        Gs.append(tmp_G)
    return Gs

def load_training_data(f_name):
    print('We are loading data from:', f_name)
    graphs = set()  # 所有边的集合 set((node1, node2))
    type_reid = dict()  # 对edge type重新编号 dict{edge_type:id}
    edge_data_by_type = dict()  # dict{edge_type:[(ndoe1, node2)]}
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = int(words[1]), int(words[2])
            if int(words[0]) not in type_reid.keys():
                type_reid[int(words[0])] = len(type_reid)
            if type_reid[int(words[0])] not in edge_data_by_type:
                edge_data_by_type[type_reid[int(words[0])]] = list()
            edge_data_by_type[type_reid[int(words[0])]].append((x, y))
            graphs.add((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Total training nodes: ' + str(len(all_nodes)))
    return list(graphs), edge_data_by_type


def load_training_data_single(f_name, main, upto):
    print('We are loading data from:', f_name)
    graphs = set()
    type_reid = dict()
    edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = int(words[1]), int(words[2])
            if int(words[0]) == main or int(words[0]) in upto:
                if int(words[0]) not in type_reid.keys():
                    type_reid[int(words[0])] = len(type_reid)
                if type_reid[int(words[0])] not in edge_data_by_type:
                    edge_data_by_type[type_reid[int(words[0])]] = list()
                edge_data_by_type[type_reid[int(words[0])]].append((x, y))
                graphs.add((x, y))
                all_nodes.append(x)
                all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Total training nodes: ' + str(len(all_nodes)))
    return list(graphs), edge_data_by_type


def load_testing_data_single(f_name, main, upto):
    print('We are loading data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_nodes = list()
    type_reid = dict()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = int(words[1]), int(words[2])
            if int(words[0]) == main or int(words[0]) in upto:
                if int(words[0]) not in type_reid.keys():
                    type_reid[int(words[0])] = len(type_reid)
                if int(words[3]) == 1:
                    if type_reid[int(words[0])] not in true_edge_data_by_type:
                        true_edge_data_by_type[type_reid[int(words[0])]] = list()
                    true_edge_data_by_type[type_reid[int(words[0])]].append((x, y))
                else:
                    if type_reid[int(words[0])] not in false_edge_data_by_type:
                        false_edge_data_by_type[type_reid[int(words[0])]] = list()
                    false_edge_data_by_type[type_reid[int(words[0])]].append((x, y))
                all_nodes.append(x)
                all_nodes.append(y)
    return true_edge_data_by_type, false_edge_data_by_type


def load_testing_data(f_name):
    print('We are loading data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_nodes = list()
    type_reid = dict()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = int(words[1]), int(words[2])
            if int(words[0]) not in type_reid.keys():
                type_reid[int(words[0])] = len(type_reid)
            if int(words[3]) == 1:
                if type_reid[int(words[0])] not in true_edge_data_by_type:
                    true_edge_data_by_type[type_reid[int(words[0])]] = list()
                true_edge_data_by_type[type_reid[int(words[0])]].append((x, y))
            else:
                if type_reid[int(words[0])] not in false_edge_data_by_type:
                    false_edge_data_by_type[type_reid[int(words[0])]] = list()
                false_edge_data_by_type[type_reid[int(words[0])]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    return true_edge_data_by_type, false_edge_data_by_type


def load_node_type(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    nodetype = dict()
    for line in data:
        nodeid, typeid = line.replace("\n", "").split(" ")
        nodetype[int(nodeid)] = str(typeid)
    return nodetype


def generate_walks(network_data, num_walks, walk_length, schema=None, nodetype=None):
    all_walks = []
    for layer_id in network_data:
        print("generating metapaths in RELATION: ", layer_id)
        tmp_data = network_data[layer_id]
        # start to do the random walk on a layer
        layer_walker = RWGraph(get_G_from_edges(tmp_data), nodetype)
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)
        all_walks.append(layer_walks)

    print('Finish generating the walks')

    return all_walks

def generate_random_walks(network_data, num_walks, walk_length):
    walker = RWGraph(get_G_from_edges(network_data), node_type=None)
    all_walks = walker.simulate_walks(num_walks, walk_length, schema=None)
    # walker = RandomWalk(get_G_from_edges(network_data), walk_length=walk_length, num_walks=num_walks, p=1, q=1,
    #                     workers=6)
    # all_walks = walker.walks
    print('Finish random walks')
    return all_walks

def generate_randomized_exploring_walks(network_data, graph_by_type, num_walks, walk_length, randomized:RandomizedFilter):
    walker = RWGraphs(get_Gs_from_edges(graph_by_type), get_G_from_edges(network_data), node_type=None)
    all_walks = walker.simulate_walks(num_walks, walk_length, schema=None)
    print('Finish random walks')
    return randomized.removeEdgeInfo(randomized.filter(all_walks))

def transform(vocab, tensor_list):
    assert len(tensor_list) > 0
    return [torch.tensor([vocab[item].index for item in tensor.cpu().numpy().tolist()]).long() for tensor in
            tensor_list]

def generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples):
    edge_type_count = len(edge_types)
    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        print('Generating neighbors for layer', r)
        g = network_data[edge_types[r]]
        for (x, y) in g:
            ix = vocab[x].index
            iy = vocab[y].index
            neighbors[ix][r].append(iy)
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:
                neighbors[i][r].extend(list(np.random.choice(neighbors[i][r], size=neighbor_samples-len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:
                neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))
    return neighbors

def original_generate_pairs(all_walks, vocab, window_size):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        for walk in walks:
            for i in range(len(walk[1])):
                for j in range(1, skip_window + 1):
                    if i - j >= 0:
                        pairs.append((vocab[walk[1][i]].index, vocab[walk[1][i - j]].index, layer_id))
                    if i + j < len(walk[1]):
                        pairs.append((vocab[walk[1][i]].index, vocab[walk[1][i + j]].index, layer_id))
    return pairs


def ori_generate_pairs(all_walks, vocab, window_size):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        for walk in walks:
            for i in range(len(walk)):
                for j in range(1, skip_window + 1):
                    if i - j >= 0:
                        pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index, layer_id))
                    if i + j < len(walk):
                        pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index, layer_id))
    return pairs


def generate_pairs(all_walks, vocab, window_size):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        for walk in walks:
            for i in range(len(walk[1])):
                for j in range(1, skip_window + 1):
                    if i - j >= 0:
                        pairs.append((vocab[walk[1][i]].index, vocab[walk[1][i - j]].index, layer_id))
                    if i + j < len(walk[1]):
                        pairs.append((vocab[walk[1][i]].index, vocab[walk[1][i + j]].index, layer_id))

    return pairs


def ori_generate_vocab(all_walks):
    index2word = []
    raw_vocab = defaultdict(int)
    for walks in all_walks:
        for walk in walks:
            for word in walk:
                raw_vocab[word] += 1
    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i

    return vocab, index2word


def to_generate_vocab(all_walks):
    index2word = []
    raw_vocab = defaultdict(int)
    for walks in all_walks:
        for walk in walks:
            for word in walk[1]:
                raw_vocab[word] += 1
    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i

    return vocab, index2word


# 通过metapath采样获得的序列和随机游走获得的序列进行结合
def hybrid_generate_vocob(all_random_walks, all_walks):
    index2word = []
    raw_vocab = defaultdict(int)
    for walks in all_walks:
        for walk in walks:
            for word in walk[1]:
                raw_vocab[word] += 1
    for walk in all_random_walks:
        for word in walk:
            raw_vocab[word] += 1
    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i

    return vocab, index2word


def list_all_nodes(whole_graph):
    nodeset = set()
    for edge in whole_graph:
        nodeset.add(edge[0])
        nodeset.add(edge[1])
    return list(nodeset)


def generate_vocab(all_walks):
    index2word = []
    raw_vocab = defaultdict(int)

    for walks in all_walks:
        for walk in walks:
            for word in walk:
                raw_vocab[word] += 1

    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i

    return vocab, index2word


def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        pass


def evaluate(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    true_num = 0
    for edge in true_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    for edge in false_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)


def to_evaluate(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    true_num = 0
    for edge in true_edges:
        tmp_score = get_score(model, edge[0], edge[1])
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1
    for edge in false_edges:
        tmp_score = get_score(model, edge[0], edge[1])
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--upto', nargs='+', type=int, default=[1,3,-4,5])

    args = parser.parse_args()
    print(args.upto)
