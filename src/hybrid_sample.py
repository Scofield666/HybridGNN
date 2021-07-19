from model import MetaGraphSager, RandomSager, RandomGraphSager
from utils import *

def prepare_samples(graphs, graphs_by_type, num_walks, random_num_walks,
                    walk_length, random_walk_length, random_sample_layers, schema, nodetype, window_size, device, filtertype=None):
    def get_neighbors(agg_neighbors):
        neighbors = dict()
        sorted_training_pairs = sorted(agg_neighbors, key=lambda x: x[2])
        for pair in sorted_training_pairs:
            key = (pair[0][0].item(), pair[1])
            if key not in neighbors.keys():
                neighbors[key] = []
            for idx, nei in enumerate(pair[0]):
                if idx == 0:
                    continue
                neighbors[key].extend([n.item() for n in nei])
        # neighbors = {k: list(set(neighbors[k])) for k, v in neighbors.items()}
        for k, v in neighbors.items():
            neighbors[k] = torch.tensor([data for data in neighbors[k]]).long()
        return neighbors

    randomizer = RandomizedFilter(schema, nodetype, filtertype)
    spec_schema_list = str(schema).split(",")
    print("starting generate meta walks")
    meta_walks = generate_walks(graphs_by_type, num_walks,
                                 walk_length, schema,
                                nodetype)
    print("starting generate random walks")
    random_walks = generate_random_walks(graphs, num_walks, walk_length)
    # random_walks = generate_randomized_exploring_walks(graphs, graphs_by_type, random_num_walks,
    #                                     random_walk_length, randomizer)
    voc, index2word = hybrid_generate_vocob(random_walks, meta_walks)
    train_pairs = generate_pairs(meta_walks, voc, window_size)

    all_nodes = list_all_nodes(whole_graph=graphs)
    print("generating schema, view...")
    # randomSager = RandomSager(graph=graphs, layers=random_sample_layers, device=device)
    randomSager = RandomGraphSager(graphs=get_Gs_from_edges(graphs_by_type), layers=random_sample_layers,
                                   scheme=schema, nodetype=nodetype, filtertype=filtertype, device=device)
    all_random_neighbors = randomSager.random_sampler(nodeids=torch.tensor(all_nodes).long())
    f_neighs = []
    agg_tensors = [torch.split(tensor, [tensor.size(0) // len(all_nodes) for _ in range(len(all_nodes))], dim=0)
                       for tensor in all_random_neighbors]
    for idx in range(len(all_nodes)):
        neighs = [agg_tensor[idx] for agg_tensor in agg_tensors]
        f_neighs.append(neighs)
    all_random_neighbors = f_neighs

    print("generating meta schema, view ...")
    f_neighs = []
    all_agg_neighbors = []
    for metaid, schema in enumerate(spec_schema_list):
        for typeid, graph in graphs_by_type.items():
            metaSager = MetaGraphSager(graph=graph, nodetype=nodetype, schema=schema, device=device)
            all_agg_neighbors.append((metaSager.meta_sampler(nodeids=torch.tensor(all_nodes).long()), typeid, metaid))

    for flatten in all_agg_neighbors:
        tensors, typid, metaid = flatten[0], flatten[1], flatten[2]
        agg_tensors = [torch.split(tensor, [tensor.size(0) // len(all_nodes) for _ in range(len(all_nodes))], dim=0)
                       for tensor in tensors]
        for idx in range(len(all_nodes)):
            neighs = [agg_tensor[idx] for agg_tensor in agg_tensors]
            f_neighs.append((neighs, typid, metaid))
    all_agg_neighbors = f_neighs
    all_agg_neighbors = [(transform(voc, all_tuple[0]), all_tuple[1], all_tuple[2]) for all_tuple in
                         all_agg_neighbors]  # [k-layer, typeid, metaid]
    # 邻居节点列表有且仅有metapath的首节点type
    meta_neighbors = get_neighbors(all_agg_neighbors)
    random_neighbors = dict()
    for neighbor in all_random_neighbors:
        if neighbor[0][0].item() not in random_neighbors.keys():
            random_neighbors[neighbor[0][0].item()] = []
        for nidx, neighbor_list in enumerate(neighbor):
            if nidx != 0:
                random_neighbors[neighbor[0][0].item()].extend([nl.item() for nl in neighbor_list])
    random_neighbors = {voc[k].index: torch.tensor([voc[nodeid].index for nodeid in v]).long() for k, v in random_neighbors.items()}
    all_nodes = [voc[nodeid].index for nodeid in all_nodes]
    return train_pairs, random_neighbors, meta_neighbors, all_nodes, len(all_nodes), index2word


if __name__ == '__main__':
    args = parse_args()
    file_name = args.input
    training_graphs, training_data_by_type = load_training_data(file_name + "/train.txt")

    config = {
        'num_walks': args.num_walks,
        'random_num_walks': args.num_walks,
        'walk_length': args.walk_length,
        'random_walk_length': 3,
        'schema': str(args.schema).replace(' ', ''),
        'nodetype': load_node_type(file_name + '/nodetype.txt'),
        'window_size': args.window_size,
    }

    prepare_samples(training_graphs, training_data_by_type, **config)
