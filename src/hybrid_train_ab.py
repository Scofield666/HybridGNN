import math
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from numpy import random
from torch.nn.parameter import Parameter
from model import MetaGraphSager, HybridGNN
from utils import *
from hybrid_sample import prepare_samples
import copy
import pickle
import os

'''
    k-neighbor, typeid, metaid
    =>
    nodeid, positive, typeid, metaid, neighbors
'''


def get_neighbors(training_pairs):
    neighbors = dict()
    sorted_training_pairs = sorted(training_pairs, key=lambda x: x[2])
    for pair in sorted_training_pairs:
        key = (pair[0][0].item(), pair[1])
        if key not in neighbors.keys():
            neighbors[key] = []
        for idx, nei in enumerate(pair[0]):
            if idx == 0:
                continue
            neighbors[key].extend([n.item() for n in nei])

    for k, v in neighbors.items():
        neighbors[k] = torch.tensor([data for data in neighbors[k]]).long()
    return neighbors


def get_hybrid_training_batches(training_pairs, batch_size, neighbors, random_neighbors, edgetypes):
    n_batches = (len(training_pairs) + (batch_size - 1)) // batch_size
    for idx in range(n_batches):
        x, y, t, n, rand_n = [], [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(training_pairs):
                break
            x.append(training_pairs[index][0])
            y.append(training_pairs[index][1])
            t.append(training_pairs[index][2])
            neigh_list = []
            for typeid in edgetypes:
                key = (training_pairs[index][0], typeid)
                neigh_list.append(neighbors[key])

            n.append(torch.stack(neigh_list, dim=0).long())
            rand_n.append(random_neighbors[training_pairs[index][0]].long())
        yield torch.tensor(x), torch.tensor(y), torch.tensor(t), torch.stack(n, dim=0).long(), torch.stack(rand_n,
                                                                                                           dim=0).long()


def get_training_batches(training_pairs, batch_size, neighbors, edgetypes, metatypes):
    n_batches = (len(training_pairs) + (batch_size - 1)) // batch_size
    for idx in range(n_batches):
        x, y, t, m, neigh, n = [], [], [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(training_pairs):
                break
            x.append(training_pairs[index][0])
            y.append(training_pairs[index][1])
            t.append(training_pairs[index][2])
            m.append(training_pairs[index][3])
            neigh_list = []
            for typeid in edgetypes:
                key = (training_pairs[index][0], typeid)
                neigh_list.append(neighbors[key])
            n.append(torch.stack(neigh_list, dim=0).long())
        yield torch.tensor(x), torch.tensor(y), torch.tensor(t), torch.tensor(m), torch.stack(n, dim=0).long()


def get_batches(training_pairs, batch_size, neighbors, edgetypes, metatypes):
    n_batches = (len(training_pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, t, m, n = [], [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(training_pairs):
                break
            sample_positive = []
            sample_positive.extend(training_pairs[index][0][1])
            sample_positive.extend(training_pairs[index][0][2])
            assert len(training_pairs[index][0]) >= 3, 'length of metapath should be larger than three.'
            x.append(training_pairs[index][0][0].item())
            y.append(random.choice(sample_positive).item())
            t.append(training_pairs[index][1])
            m.append(training_pairs[index][2])
            neigh_list = []
            for typeid in edgetypes:
                key = (x[0], typeid)
                neigh_list.append(neighbors[key])
            n.append(torch.stack(neigh_list, dim=0).long())
        yield torch.tensor(x).long(), torch.tensor(y).long(), torch.tensor(t).long(), \
              torch.tensor(m).long(), torch.stack(n, dim=0).long()


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, device, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weights = F.normalize(
            torch.tensor(
                [
                    (math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)
                    for k in range(num_nodes)
                ]
            ),
            dim=0,
        )

        self.reset_parameters()
        self.device = device

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1))
        )
        negs = torch.multinomial(
            self.sample_weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs]).to(self.device)
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n


'''
    @network_data: 多种类型的图
    @graphs: sparse_graph，同样是多种类型的图
    @schema_list: meta集合
    @nodetype: 节点集合
'''


def train_model(graph, graph_by_type, args):
    # # 随机采样, 按每种relation
    # random_all_walks = generate_walks(network_data, args.num_walks, args.walk_length, None, None)
    # # Metapath采样
    # schemes = scheme_list.split(",")
    # all_walks = generate_walks(network_data, args.num_walks, args.walk_length, args.schema, nodetype)
    # vocab, index2word = hybrid_generate_vocob(all_walks, random_all_walks)
    # train_pairs = generate_pairs(all_walks, vocab, args.window_size, schemes, nodetype)
    # # 聚合 生成k阶邻居
    # all_nodes = list_all_nodes(all_walks)
    # all_agg_neighbors = []
    # print("generating schema, view...")
    # for metaid, schema in enumerate(schemes):
    #     for typeidx, (typeid, graph) in enumerate(network_data.items()):
    #         metaSager = MetaGraphSager(graph=graph, nodetype=nodetype, schema=schema)
    #         all_agg_neighbors.append((metaSager.meta_sampler(nodeids=torch.tensor(all_nodes).long()), typeidx, metaid))
    # f_neighs = []
    # for flatten in all_agg_neighbors:
    #     tensors, typidx, metaid = flatten[0], flatten[1], flatten[2]
    #     agg_tensors = [torch.split(tensor, [tensor.size(0) // len(all_nodes) for _ in range(len(all_nodes))], dim=0)
    #                    for tensor in tensors]
    #     for idx in range(len(all_nodes)):
    #         neighs = [agg_tensor[idx] for agg_tensor in agg_tensors]
    #         f_neighs.append((neighs, typidx, metaid))
    # all_agg_neighbors = f_neighs
    # # vocab, index2word = to_generate_vocab([all_tuple[0] for all_tuple in all_agg_neighbors])
    # all_agg_neighbors = [(transform(vocab, all_tuple[0]), all_tuple[1], all_tuple[2]) for all_tuple in
    #                      all_agg_neighbors]  # [k-layer, typeid, metaid]
    #
    # all_nodes = list(set(vocab[node].index for node in all_nodes))  # reid
    # metatypes = list(set([p[2] for p in all_agg_neighbors]))
    # edgetypes = list(set([p[1] for p in all_agg_neighbors]))
    #
    # # 邻居节点列表有且仅有metapath的首节点type
    # neighbors = get_neighbors(all_agg_neighbors)
    # # edge_types = list(network_data.keys())
    # num_nodes = len(index2word)
    # edge_type_count = len(edgetypes)
    # epochs = args.epoch
    # batch_size = args.batch_size
    # embedding_size = args.dimensions
    # num_sampled = args.negative_samples
    #
    # random_neighbors = generate_neighbors(network_data, vocab, num_nodes, edgetypes, neighbor_samples=5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pickle_name = '_'.join([HybridGNN.name, str(args.input).split("/")[-1], str(args.num_walks), str(args.walk_length),
                   str(args.random_walk_length), str(args.random_sample_layers),
                   str(args.schema).replace(' ', ''), str(args.window_size), str(args.percent), '.pkl'])
    if os.path.exists(pickle_name):
        print("DIRECTLY loading pickle files: %s...." % pickle_name)
        with open(pickle_name, 'rb') as f:
            pickle_data = pickle.load(f)
        train_pairs = pickle_data['train_pairs']
        random_neighbors = pickle_data['random_neighbors']
        meta_neighbors = pickle_data['meta_neighbors']
        all_nodes = pickle_data['all_nodes']
        num_nodes = pickle_data['num_nodes']
        index2word = pickle_data['index2word']
    else:
        print("GENERATING pickle files: %s...." % pickle_name)
        train_pairs, random_neighbors, meta_neighbors, \
        all_nodes, num_nodes, index2word = prepare_samples(graph, graph_by_type, **{
            'num_walks': args.num_walks,
            'random_num_walks': args.num_walks,
            'walk_length': args.walk_length,
            'random_walk_length': args.random_walk_length,
            'random_sample_layers': args.random_sample_layers,
            'schema': str(args.schema).replace(' ', ''),
            'nodetype': load_node_type(file_name + '/nodetype.txt'),
            'window_size': args.window_size,
            'device': device,
            'filtertype': args.filter,
        })
        with open(pickle_name, 'wb') as f:
            pickle.dump({'train_pairs': train_pairs,
                         'random_neighbors': random_neighbors,
                         'meta_neighbors': meta_neighbors,
                         'all_nodes': all_nodes,
                         'num_nodes': num_nodes,
                         'index2word': index2word}, f)
    edgetypes = [i for i in range(args.edge_type_count)]

    model = HybridGNN(max_users=num_nodes, edge_type_count=args.edge_type_count, schema=args.schema, edge_dim=args.edge_dim,
                      input_dim=args.dimensions, output_dim=args.dimensions, random_layer=args.random_sample_layers)

    nsloss = NSLoss(num_nodes, args.negative_samples, device, args.dimensions)

    model.to(device)
    nsloss.to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": nsloss.parameters()}], lr=args.lr
    )
    best_avg_auc = .0
    best_avg_pr = .0
    best_avg_f1 = .0
    best_test_avg_auc = .0
    best_test_avg_pr = .0
    best_test_avg_f1 = .0
    best_score = 0
    patience = 0
    # print("starting to train from epochs.")
    for epoch in range(args.epoch):
        random.shuffle(train_pairs)
        batches = get_hybrid_training_batches(train_pairs, args.batch_size, meta_neighbors,
                                              random_neighbors, edgetypes)

        data_iter = tqdm.tqdm(
            batches,
            desc="epoch %d" % (epoch),
            total=(len(train_pairs) + (args.batch_size - 1)) // args.batch_size,
            bar_format="{l_bar}{r_bar}",
        )
        avg_loss = 0.0
        # nodes, positive, typeid, metaid, neighbors
        for i, data in enumerate(data_iter):
            nodeid, connid, typeid, meta_neigh, rand_neigh = data
            optimizer.zero_grad()
            rows = torch.tensor([i for i in range(len(nodeid))]).long().to(device)
            embs = model(nodeid.to(device), (rows, typeid.to(device)),
                         meta_neigh.to(device), rand_neigh.to(device))
            loss = nsloss(nodeid.to(device), embs, connid.to(device))
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if i % 50000 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.write(str(post_fix))

        final_model = dict(zip(edgetypes, [dict() for _ in range(args.edge_type_count)]))
        for i in range(len(all_nodes)):
            train_inputs = torch.tensor([i for _ in range(args.edge_type_count)]).long().to(device)
            train_types = torch.tensor(list(range(args.edge_type_count))).long().to(device)
            node_neigh = []
            for j in range(len(edgetypes)):
                key = (i, j)
                node_neigh.append(meta_neighbors[key])
            node_neigh = torch.stack(node_neigh, dim=0).to(device)
            node_neigh = torch.stack(
                [node_neigh for _ in range(args.edge_type_count)], dim=0
            ).long().to(device)
            rand_node_neigh = torch.stack(
                [random_neighbors[i] for _ in range(args.edge_type_count)], dim=0
            ).long().to(device)
            rows = torch.tensor([i for i in range(train_inputs.size(0))]).long().to(device)
            # node_emb = model(train_inputs, (rows, train_types), node_neigh)
            node_emb = model(train_inputs, (rows, train_types), node_neigh, rand_node_neigh)
            for j in range(args.edge_type_count):
                final_model[edgetypes[j]][index2word[i]] = (
                    node_emb[j].cpu().detach().numpy()
                )

        valid_aucs, valid_f1s, valid_prs = [], [], []
        test_aucs, test_f1s, test_prs = [], [], []
        for i in range(args.edge_type_count):
            if args.eval_type == "all" or edgetypes[i] in args.eval_type.split(","):
                tmp_auc, tmp_f1, tmp_pr = to_evaluate(
                    final_model[edgetypes[i]],
                    valid_true_data_by_edge[edgetypes[i]],
                    valid_false_data_by_edge[edgetypes[i]],
                )
                valid_aucs.append(tmp_auc)
                valid_f1s.append(tmp_f1)
                valid_prs.append(tmp_pr)

                tmp_auc, tmp_f1, tmp_pr = to_evaluate(
                    final_model[edgetypes[i]],
                    testing_true_data_by_edge[edgetypes[i]],
                    testing_false_data_by_edge[edgetypes[i]],
                )
                test_aucs.append(tmp_auc)
                test_f1s.append(tmp_f1)
                test_prs.append(tmp_pr)
        # print("valid auc:", np.mean(valid_aucs))
        # print("valid pr:", np.mean(valid_prs))
        # print("valid f1:", np.mean(valid_f1s))

        average_auc = np.mean(test_aucs)
        average_f1 = np.mean(test_f1s)
        average_pr = np.mean(test_prs)
        if best_test_avg_auc < average_auc:
            best_test_avg_auc = average_auc
            best_test_avg_pr = average_pr
            best_test_avg_f1 = average_f1
        # print("test auc:", average_auc)
        # print("test pr:", average_pr)
        # print("test f1:", average_f1)
        cur_score = np.mean(valid_aucs)
        if cur_score > best_score:
            # print(f"best valid auc improve from {best_score} to {cur_score}.")
            best_avg_auc = np.mean(valid_aucs)
            best_avg_pr = np.mean(valid_prs)
            best_avg_f1 = np.mean(valid_f1s)
            best_score = cur_score
            patience = 0
            torch.save(copy.deepcopy(model.state_dict()), model.name + "_" + args.input.split("/")[-1] + '.chkpt')
            model.load_state_dict(torch.load(model.name + "_" + args.input.split("/")[-1] + '.chkpt'))

        else:
            patience += 1
            if patience > args.patience:
                print("Early Stopping")
                # print("current best valid/test auc: {}, {}".format(best_avg_auc, best_test_avg_auc))
                # print("current best valid/test pr: {}, {}".format(best_avg_pr, best_test_avg_pr))
                # print("current best valid/test f1: {}, {}".format(best_avg_f1, best_test_avg_f1))
                break

        # print("current best valid/test auc: {}, {}".format(best_avg_auc, best_test_avg_auc))
        # print("current best valid/test pr: {}, {}".format(best_avg_pr, best_test_avg_pr))
        # print("current best valid/test f1: {}, {}".format(best_avg_f1, best_test_avg_f1))
    print("current best valid/test auc: {}, {}".format(best_avg_auc, best_test_avg_auc))
    print("current best valid/test pr: {}, {}".format(best_avg_pr, best_test_avg_pr))
    print("current best valid/test f1: {}, {}".format(best_avg_f1, best_test_avg_f1))
    return average_auc, average_f1, average_pr


def load_node_type(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    nodetype = dict()
    for line in data:
        nodeid, typeid = line.replace("\n", "").split(" ")
        nodetype[int(nodeid)] = str(typeid)
    return nodetype


if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)
    # preprocess
    schemas = args.schema.replace(' ', '')
    train_txt = '/train'
    valid_txt = '/valid'
    test_txt = '/test'
    if args.percent == 0:
        train_txt = train_txt + ".txt"
        valid_txt = valid_txt + ".txt"
        test_txt = test_txt + ".txt"
    else:
        train_txt = train_txt + "_" + str(args.percent) + ".txt"
        valid_txt = valid_txt + "_" + str(args.percent) + ".txt"
        test_txt = test_txt + "_" + str(args.percent) + ".txt"
    training_graphs, training_data_by_type = load_training_data(file_name + train_txt)
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(
        file_name + valid_txt
    )
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(
        file_name + test_txt
    )
    nodetype = load_node_type(file_name + '/nodetype.txt')

    average_auc, average_f1, average_pr = train_model(training_graphs, training_data_by_type, args)

    # print("Overall ROC-AUC:", average_auc)
    # print("Overall PR-AUC", average_pr)
    # print("Overall F1:", average_f1)

# 检查metapath采样合理性、检查training pair生成合理性
