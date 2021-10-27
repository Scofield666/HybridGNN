import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from aggregator import MeanAggregator
from sampler import SparseUniformNeighborSampler, SparseUniformNeighborMetaSampler, UniformNeighborMetaSampler, UniformRandomSampler
import math

class RandomGraphSager:
    def __init__(self, graphs, layers, scheme, nodetype, filtertype, device):
        print("Using filter type : ", filtertype)
        self.graphs = graphs
        self.layers = layers
        self.nodetype = nodetype
        self.scheme_candidates = [s.replace("-", str(i)) for i in range(len(graphs)) for s in scheme.split(",")]
        self.sample_numbers = [2 * i + 3 for i in range(self.layers - 1)]
        self.sampler_fn = UniformRandomSampler(self.graphs, nodetype, filtertype, self.scheme_candidates, self.sample_numbers)
        self.device = device

    def random_sampler(self, nodeids):
        all_feats = [nodeids]
        filter_list = [self.nodetype[n.item()] for n in nodeids]
        for idx, neighbor_nodes in enumerate(range(self.layers - 1)):
            nodeids, filter_list = self.sampler_fn(ids=nodeids, filter_list=filter_list, index=idx + 1, n_samples=self.sample_numbers[idx])
            nodeids = nodeids.contiguous().view(
                -1).to(self.device)
            all_feats.append(nodeids)
        return all_feats


class RandomSager:
    def __init__(self, graph, layers, device):
        self.graph = graph
        self.layers = layers
        self.sample_numbers = [2 * i + 3 for i in range(self.layers - 1)]
        self.sampler_fn = UniformNeighborMetaSampler(self.graph, schema=None, nodetpye=None)
        self.device = device

    def random_sampler(self, nodeids):
        all_feats = [nodeids]
        for idx, neighbor_nodes in enumerate(range(self.layers - 1)):
            nodeids = self.sampler_fn(ids=nodeids, index=idx + 1, n_samples=self.sample_numbers[idx]).contiguous().view(
                -1).to(self.device)
            all_feats.append(nodeids)
        return all_feats


class MetaGraphSager:
    # schema: U-A-I-A-U
    def __init__(self, graph, nodetype, schema, device):
        self.graph = graph
        self.schema_list = schema.split("-")
        self.sample_numbers = [2 * i + 3 for i in range(len(self.schema_list) - 1)]
        # sampler
        self.sampler_fn = UniformNeighborMetaSampler(self.graph, schema=self.schema_list, nodetpye=nodetype)
        self.device = device

    def meta_sampler(self, nodeids):
        all_feats = [nodeids]
        for idx, neighbor_nodes in enumerate(range(len(self.schema_list) - 1)):
            nodeids = self.sampler_fn(ids=nodeids, index=idx + 1, n_samples=self.sample_numbers[idx]).contiguous().view(
                -1).to(self.device)
            all_feats.append(nodeids)

        return all_feats


class MetaSager:
    # schema: U-A-I-A-U
    def __init__(self, graph: csr_matrix, nodetype, schema, device):
        self.graph = graph
        self.schema_len = schema.split("-")
        self.sample_numbers = [2 * i + 3 for i in range(len(self.schema_len) - 1)]
        # sampler
        self.sampler_fn = SparseUniformNeighborMetaSampler(self.graph, schema=schema, nodetpye=nodetype)
        self.device = device

    def meta_sampler(self, nodeids):
        all_feats = [nodeids]
        for idx, neighbor_nodes in enumerate(len(self.schema_len) - 1):
            nodeids = self.sampler_fn(ids=nodeids, index=idx + 1, n_samples=self.sample_numbers[idx]).contiguous().view(
                -1).to(self.device)
            all_feats.append(nodeids)

        return all_feats


class GraphSage(nn.Module):
    def __init__(self, graph: csr_matrix, layers, input_dim, output_dim):
        super(GraphSage, self).__init__()
        # Parameters
        self.layers = layers
        self.graph = graph
        self.sample_numbers = [2 * i + 3 for i in range(self.layers)]

        # sampler
        self.sampler_fn = SparseUniformNeighborSampler(self.graph)
        # Network
        self.user_embedding = nn.Embedding(graph.shape[0], input_dim)
        # print(">>>>>>>>>", sum([x[1].nelement() for x in self.user_embedding.named_parameters()]))
        agg_layers = []
        for spec_layer in range(self.layers):
            agg = MeanAggregator(input_dim=input_dim, output_dim=input_dim // 2)
            #   print(">>>>>>>>>", sum([x[1].nelement() for x in agg.named_parameters()]))
            agg_layers.append(agg)
        self.agg_layers = nn.Sequential(*agg_layers)
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.param_reset()

    def param_reset(self):
        self.user_embedding.weight.data.uniform_(-1.0, 1.0)

    def forward(self, nodeids):
        node_embed = self.user_embedding(nodeids)
        all_feats = [node_embed]
        for idx, neighbor_nodes in enumerate(range(self.layers)):
            nodeids = self.sampler_fn(ids=nodeids, n_samples=self.sample_numbers[idx]).contiguous().view(-1)
            all_feats.append(self.user_embedding(nodeids))
        for agg_layer in self.agg_layers.children():
            all_feats = [agg_layer(all_feats[k], all_feats[k + 1]) for k in range(len(all_feats) - 1)]

        assert len(all_feats) == 1, "asseration len(all_feats) == 1"
        out = F.normalize(all_feats[0], dim=1)
        return self.fc(out)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn


class RegLoss():
    def __init__(self, decay):
        self.decay = decay

    def __call__(self, embedding):
        return self.decay * torch.mean(embedding.pow(2))


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, device, embedding_size=200):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.context_embedding = nn.Embedding(num_nodes, embedding_size)
        self.sample_weights = F.normalize(
            torch.tensor(
                [
                    (math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)
                    for k in range(num_nodes)
                ]
            ),
            dim=0
        )
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        self.context_embedding.weight.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, embed, pos_neighbors):
        n = embed.size(0)
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embed, self.context_embedding(pos_neighbors)), 1))
        )
        negs = torch.multinomial(
            self.sample_weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled).to(self.device)
        noise = torch.neg(self.context_embedding(negs)).to(self.device)
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, embed.unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n


class KGATNE(nn.Module):
    name = "kgatne"

    def __init__(self, graphs, edge_type_count, layers=4, edge_dim=100, input_dim=200, output_dim=200):
        super(KGATNE, self).__init__()
        # Param
        self.max_users = max([g.shape[0] for g in graphs])
        self.layers = layers
        self.graph_types = len(graphs)
        self.alpha = 0.1
        self.embedding_size = input_dim
        self.edge_type_count = edge_type_count
        self.reflect = nn.Parameter(torch.FloatTensor(self.edge_type_count, edge_dim, output_dim))

        # Network
        graph_sager = []
        self.base_embed = nn.Embedding(self.max_users, input_dim)
        for graph in graphs:
            sager = GraphSage(graph=graph, layers=layers, input_dim=edge_dim, output_dim=edge_dim)
            graph_sager.append(sager)

        self.graph_sager = nn.ModuleList(graph_sager)
        self.slf_attn = MultiHeadAttention(n_head=1, d_model=edge_dim, d_k=edge_dim, d_v=edge_dim)

        self.param_reset()
        # print(">>>>>>>>> base", sum([x[1].nelement() for x in self.base_embed.named_parameters()]))
        # print(">>>>>>>>> graph_sager", sum([x[1].nelement() for x in self.graph_sager.named_parameters()]))
        # print(">>>>>>>>> slf_attn", sum([x[1].nelement() for x in self.slf_attn.named_parameters()]))
        # print(">>>>>>>>> proj", sum([x[1].nelement() for x in self.proj.named_parameters()]))

    def param_reset(self):
        self.base_embed.weight.data.uniform_(-1.0, 1.0)
        self.reflect.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, nodeids, edgetypes):
        base_embedding = self.base_embed(nodeids)
        spec_embed_list = []
        for cur_type in range(self.graph_types):
            embed = self.graph_sager[cur_type](nodeids)
            spec_embed_list.append(embed)
        spec_embed = torch.stack(spec_embed_list, dim=1)
        spec_embed, _ = self.slf_attn(spec_embed, spec_embed, spec_embed)
        out_embed = torch.matmul(
            spec_embed[list(zip(*[(i, edgetype) for i, edgetype in enumerate(edgetypes)]))].unsqueeze(dim=1),
            self.reflect[edgetypes]).squeeze(dim=1)

        return F.normalize(base_embedding + out_embed, dim=1)


class KGATNEOld(nn.Module):
    name = "kgatne"

    def __init__(self, max_users, edge_type_count, layers=4, edge_dim=100, input_dim=200, output_dim=200):
        super(KGATNEOld, self).__init__()
        # Param
        self.layers = layers
        self.alpha = 0.1
        self.embedding_size = input_dim
        self.max_users = max_users
        self.edge_type = edge_type_count

        # Network
        self.reflect = nn.Parameter(torch.FloatTensor(edge_type_count, edge_dim, output_dim))
        self.base_embed = nn.Embedding(self.max_users, input_dim)
        # self.type_embed = nn.Embedding(self.max_users, edge_dim)  # bug
        self.type_embed = nn.Parameter(torch.FloatTensor(self.max_users, edge_type_count, edge_dim))
        self.slf_attn = MultiHeadAttention(n_head=1, d_model=edge_dim, d_k=edge_dim, d_v=edge_dim)
        self.param_reset()

    def param_reset(self):
        self.base_embed.weight.data.uniform_(-1.0, 1.0)
        self.reflect.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.type_embed.data.uniform_(-1.0, 1.0)

    def forward(self, nodeids, edgetype, neighbors):
        base_embedding = self.base_embed(nodeids)
        spec_embed_list = []
        nei_embed = self.type_embed[neighbors]
        # nei_embed = self.type_embed(neighbors)
        for i in range(nei_embed.size(1)):
            agg_embed = torch.sum(nei_embed[:, i, :, i, :], dim=1)
            spec_embed_list.append(agg_embed)
        spec_embed = torch.stack(spec_embed_list, dim=1)
        spec_embed, _ = self.slf_attn(spec_embed, spec_embed, spec_embed)
        out_embed = torch.matmul(spec_embed[edgetype[0], edgetype[1]].unsqueeze(dim=1),
                                 self.reflect[edgetype[1]]).squeeze(dim=1)
        return F.normalize(base_embedding + out_embed, dim=1)


class MetaGNN(nn.Module):
    name = "MetaGNN"

    '''
        schema: "U-I-U, U-A-I-A-U" 
    '''

    def __init__(self, max_users, edge_type_count, schema=None, edge_dim=100, input_dim=200, output_dim=200):
        super(MetaGNN, self).__init__()
        # Param
        self.alpha = 0.1
        self.schema_types = schema.replace(' ', '').split(",")
        self.embedding_size = input_dim
        self.edge_dim = edge_dim
        self.max_users = max_users
        self.edge_type = edge_type_count
        self.partition_list = self.get_partition_list()

        meta_agg_layers = []
        for idx in range(len(self.schema_types)):
            agg_layers = []
            for spec_layer in range(len(self.schema_types[idx].split("-")) - 1):
                agg = MeanAggregator(input_dim=edge_dim, output_dim=edge_dim // 2)
                agg_layers.append(agg)
            self.agg_layers = nn.Sequential(*agg_layers)
            meta_agg_layers.append(self.agg_layers)
        self.meta_agg_layer = nn.Sequential(*meta_agg_layers)
        self.fc = nn.Linear(input_dim, output_dim, bias=True)

        # Network
        # self.reflect = nn.Parameter(torch.FloatTensor(len(self.schema_types), edge_type_count, edge_dim, output_dim))
        self.reflect = nn.Parameter(torch.FloatTensor(edge_type_count, edge_dim, output_dim))
        self.base_embed = nn.Embedding(self.max_users, input_dim)
        # self.type_embed = nn.Embedding(self.max_users, edge_dim)  # bug
        self.type_embed = nn.Parameter(
            torch.FloatTensor(self.max_users, edge_type_count, len(self.schema_types), edge_dim))
        self.slf_meta_attn = MultiHeadAttention(n_head=1, d_model=edge_dim, d_k=edge_dim, d_v=edge_dim)
        self.slf_view_attn = MultiHeadAttention(n_head=1, d_model=edge_dim, d_k=edge_dim, d_v=edge_dim)
        self.param_reset()

    def param_reset(self):
        self.base_embed.weight.data.uniform_(-1.0, 1.0)
        self.reflect.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.type_embed.data.uniform_(-1.0, 1.0)

    def get_partition_list(self):
        schema_len_list = []
        for schema in self.schema_types:
            schema_len_list.append(len(schema.split("-")))
        partition_list = []
        for idx in range(len(schema_len_list)):
            tmp = []
            for ilayer in range(1, schema_len_list[idx]):
                start = 1
                for num in range(2 * ilayer + 1, 0, -2):
                    start *= num
                tmp.append(start)
            partition_list.append(tmp)
        return partition_list

    '''
        multiple views
        multiple schemas defined on view
        ==> variable length on the same view
        ?
        How to learn the path info from aggregation?
        Node: k-layer aggregation.
            different views => node with k-layer neighbors  # * types * (k * N)
            different schemas => node with variable-layer neighbors 
                # * types * (k1 * N)
                # * types * (k2 * N)
                # * types * (k3 * N)
                => (# * types * [(k1 + k2 + k3) * N])
                
                k  : 1 + 1 * 3 + 1 * 5
                k1 : 1 + 1 * 3
                k2 : 1 + 1 * 5 + 1 * 7
                k3 : 1 + 1 * 5 + 1 * 7
                
            Kn: schemas 
            K1: U-I-U, K2: U-A-I-A-U, K3: U-I-A-I-U
            K1: (# + # * 3 + # * 5)
            K2: (# + # * 3 + # * 5 + # * 7 + # * 9)
            K3: (# + # * 3 + # * 5 + # * 7 + # * 9)
            
            # overall
            # * types * [(# * 3 + # * 5), (# * 3 + # * 5 + # * 7 + # * 9),
                (# * 3 + # * 5 + # * 7 + # * 9)]
            
            e.g.
            # 64 * type * (1 + 1 * 3 + 1 * 3 * 5)
            # 64 * type * 1
            # 64 * type * (1 * 3)
            # 64 * type * (1 * 3 * 5)
            # ======> 按type拆解
            # 64 * 1, 64 * (1*3), 64 * (1 * 3 * 5)
            # 64 * 200, (64 * 3) * 200, (64 * 3 * 5) * 200
            # 64 * 200
    '''

    def forward(self, nodeids, edgetype, neighbors):
        base_embedding = self.base_embed(nodeids)
        neighbors = list(torch.split(neighbors, [sum(overall) for overall in self.partition_list], dim=2))
        # batch * types * (# * 3 + # * 5), batch * types * (# * 3 + # * 5 + # * 7 + # * 9) ....
        # loop schema
        meta_view = []
        for idx, neigh in enumerate(neighbors):
            all_feats = []  # (self.max_users, edge_type_count, edge_dim))
            base = self.type_embed[nodeids]  # batch * type * embed
            # batch * type * meta * embed
            layer_neigh = list(neighbors[idx].split(self.partition_list[idx], dim=2))
            # batch * types * (batch * 3), batch * types * (batch * 3 * 5)
            for neigh in layer_neigh:
                all_feats.append(self.type_embed[neigh][:, :, :, :, idx, :])  # batch * type * num * type * embed
                # batch * type * num * type * meta * dim
            spec_embed_list = []
            for typeid in range(base.size(1)):
                # (batch * num) * embed
                type_feats = [feats[:, typeid, :, typeid, :].contiguous().view(-1, self.edge_dim) for feats in
                              all_feats]
                type_feats.insert(0, base[:, typeid, idx, :])
                # print(len(self.meta_agg_layer[idx].children()))
                for agg_layer in self.meta_agg_layer[idx].children():
                    type_feats = [agg_layer(type_feats[k], type_feats[k + 1]) for k in range(len(type_feats) - 1)]
                assert len(type_feats) == 1, "asseration len(all_feats) == 1"
                spec_embed_list.append(type_feats[0])  # batch * embed
            # [] len: types, batch * embed
            spec_embed = torch.stack(spec_embed_list, dim=1)  # batch * type * embed
            meta_view.append(spec_embed)

        # self attention on type
        meta_view = [self.slf_view_attn(view, view, view)[0] for view in meta_view]
        meta_view = torch.stack(meta_view, dim=1)  # batch * meta * type * embed embeding
        # type embedding
        # self attention on meta; [] len: typeid, batch * meta * embed
        meta_view = [self.slf_meta_attn(meta_view[:, :, typeid, :],
                                        meta_view[:, :, typeid, :],
                                        meta_view[:, :, typeid, :])[0] for typeid in range(meta_view.size(2))]
        meta_view = torch.stack(meta_view, dim=2)  # batch * meta * type * embed

        # batch * embed
        # out_embed = torch.matmul(meta_view[edgetype[0], metatype, edgetype[1]].unsqueeze(dim=1),
        #                          self.reflect[metatype, edgetype[1]]).squeeze(dim=1)
        out_embed = torch.matmul(torch.mean(meta_view, dim=1)[edgetype[0], edgetype[1]].unsqueeze(dim=1),
                                 self.reflect[edgetype[1]]).squeeze(dim=1)
        return F.normalize(base_embedding + out_embed, dim=1)


class HybridGNN(nn.Module):
    name = "hybrid"

    '''
        schema: "U-I-U, U-A-I-A-U" 
    '''

    def __init__(self, max_users, edge_type_count, schema=None, edge_dim=100, input_dim=200, output_dim=200,
                 random_layer=3, att_vis=False):
        super(HybridGNN, self).__init__()
        # Param
        self.alpha = 0.1
        self.schema_types = schema.replace(' ', '').split(",")
        self.embedding_size = input_dim
        self.edge_dim = edge_dim
        self.max_users = max_users
        self.edge_type = edge_type_count
        self.partition_list = self.get_partition_list()
        self.random_partition_list = self.get_random_partition_list(random_layer)
        self.random_sample_layer = random_layer
        self.att_vis = att_vis

        # hybrid mixture
        self.node_type_embeddings = nn.Parameter(
            torch.FloatTensor(self.max_users, edge_type_count, edge_dim)
        )

        meta_agg_layers = []
        for idx in range(len(self.schema_types)):
            agg_layers = []
            for spec_layer in range(len(self.schema_types[idx].split("-")) - 1):
                agg = MeanAggregator(input_dim=edge_dim, output_dim=edge_dim // 2)
                agg_layers.append(agg)
            self.agg_layers = nn.Sequential(*agg_layers)
            meta_agg_layers.append(self.agg_layers)
        self.meta_agg_layer = nn.Sequential(*meta_agg_layers)
        self.random_agg_layer = nn.Sequential(*[
            MeanAggregator(input_dim=edge_dim, output_dim=edge_dim // 2)
            for _ in range(self.random_sample_layer - 1)
        ])
        self.fc = nn.Linear(input_dim, output_dim, bias=True)

        # Network
        # self.reflect = nn.Parameter(torch.FloatTensor(len(self.schema_types), edge_type_count, edge_dim, output_dim))
        self.reflect = nn.Parameter(torch.FloatTensor(edge_type_count, edge_dim, output_dim))
        self.base_embed = nn.Embedding(self.max_users, input_dim)
        self.rw_embed = nn.Parameter(
            torch.FloatTensor(self.max_users, edge_dim)
        )
        self.type_embed = nn.Parameter(
            torch.FloatTensor(self.max_users, edge_type_count, len(self.schema_types), edge_dim))
        self.slf_meta_attn = MultiHeadAttention(n_head=1, d_model=edge_dim, d_k=edge_dim, d_v=edge_dim)
        self.slf_view_attn = MultiHeadAttention(n_head=1, d_model=edge_dim, d_k=edge_dim, d_v=edge_dim)
        self.param_reset()

    def param_reset(self):
        self.base_embed.weight.data.uniform_(-1.0, 1.0)
        self.type_embed.data.uniform_(-1.0, 1.0)
        self.rw_embed.data.uniform_(-1.0, 1.0)
        self.reflect.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def get_random_partition_list(self, random_layers):
        random_partition_list = []
        for ilayer in range(1, random_layers):
            start = 1
            for num in range(2 * ilayer + 1, 0, -2):
                start *= num
            random_partition_list.append(start)
        return random_partition_list

    def get_partition_list(self):
        schema_len_list = []
        for schema in self.schema_types:
            schema_len_list.append(len(schema.split("-")))
        partition_list = []
        for idx in range(len(schema_len_list)):
            tmp = []
            for ilayer in range(1, schema_len_list[idx]):
                start = 1
                for num in range(2 * ilayer + 1, 0, -2):
                    start *= num
                tmp.append(start)
            partition_list.append(tmp)
        return partition_list

    def forward(self, nodeids, edgetype, neighbors, random_neighbors):
        base_embedding = self.base_embed(nodeids)
        neighbors = list(torch.split(neighbors, [sum(overall) for overall in self.partition_list], dim=2))
        # batch * types * (# * 3 + # * 5), batch * types * (# * 3 + # * 5 + # * 7 + # * 9) ....
        # loop schema
        meta_view = []
        for idx, neigh in enumerate(neighbors):
            all_feats = []  # (self.max_users, edge_type_count, edge_dim))
            base = self.type_embed[nodeids]  # batch * type * embed
            # batch * type * meta * embed
            layer_neigh = list(neighbors[idx].split(self.partition_list[idx], dim=2))
            # batch * types * (batch * 3), batch * types * (batch * 3 * 5)
            for neigh in layer_neigh:
                all_feats.append(self.type_embed[neigh][:, :, :, :, idx, :])  # batch * type * num * type * embed
                # batch * type * num * type * meta * dim
            spec_embed_list = []
            for typeid in range(base.size(1)):
                # (batch * num) * embed
                type_feats = [feats[:, typeid, :, typeid, :].contiguous().view(-1, self.edge_dim) for feats in
                              all_feats]
                type_feats.insert(0, base[:, typeid, idx, :])
                # print(len(self.meta_agg_layer[idx].children()))
                for agg_layer in self.meta_agg_layer[idx].children():
                    type_feats = [agg_layer(type_feats[k], type_feats[k + 1]) for k in range(len(type_feats) - 1)]
                assert len(type_feats) == 1, "asseration len(all_feats) == 1"
                spec_embed_list.append(type_feats[0])  # batch * embed
            # [] len: types, batch * embed
            spec_embed = torch.stack(spec_embed_list, dim=1)  # batch * type * embed
            meta_view.append(spec_embed)

        if random_neighbors is not None:
            all_feats = []  # (self.max_users, edge_type_count, edge_dim))
            layer_neigh = list(random_neighbors.split(self.random_partition_list, dim=1))  # B * 3, B * (3*5)
            for neigh in layer_neigh:
                all_feats.append(self.rw_embed[neigh].contiguous().view(-1, self.edge_dim))  # batch * embed
            all_feats.insert(0, self.rw_embed[nodeids])
            # print(len(self.meta_agg_layer[idx].children()))
            for agg_layer in self.random_agg_layer.children():
                all_feats = [agg_layer(all_feats[k], all_feats[k + 1]) for k in range(len(all_feats) - 1)]
            assert len(all_feats) == 1, "asseration len(all_feats) == 1"
            random_view = all_feats[0].unsqueeze(dim=1)  # [N, 1, dim]
            # -------------------------------------- modify --------------------------------------#
            # meta_view = [torch.cat([view, random_view], dim=1) for view in meta_view]
            # -------------------------------------- modify --------------------------------------#
            meta_view = torch.stack(meta_view, dim=1)  # [N, schema_num, edge_type_cnt, dim]
            random_view = random_view.repeat(1, meta_view.size(2), 1).unsqueeze(1)  # [N, 1, edge_type_cnt, dim]
            meta_view = torch.cat((meta_view, random_view), dim=1)  # [N, schema_num+1, edge_type_cnt, dim]
            meta_view = [meta_view[:, meta_id, :, :] for meta_id in range(meta_view.size(1))]  # list[tensor[], tensor[]] tensor shape:[N, edge_type_cnt, dim]
            # radnom hybrid end

        # self attention on type
        meta_view = [self.slf_view_attn(view, view, view)[0] for view in meta_view]
        meta_view = torch.stack(meta_view, dim=1)  # batch * meta * type * embed embeding  [N, schema_num+1, edge_type_cnt, dim]
        # # type embedding
        # # self attention on meta; [] len: typeid, batch * meta * embed
        # meta_view = [self.slf_meta_attn(meta_view[:, :, typeid, :],
        #                                 meta_view[:, :, typeid, :],
        #                                 meta_view[:, :, typeid, :])[0] for typeid in range(meta_view.size(2))]
        # meta_view = torch.stack(meta_view, dim=2)  # batch * meta * type * embed

        # ----------------------- attention score visualization ----------------------- #
        meta_view_att_scores = []
        meta_view_tmp = []
        for typeid in range(meta_view.size(2)):
            att_res, att_score = self.slf_meta_attn(meta_view[:, :, typeid, :],
                                                    meta_view[:, :, typeid, :],
                                                    meta_view[:, :, typeid, :])  # res:[N, schema_num+1, dim]  att_score [N, n_head, schema_num+1, schema_num+1]
            meta_view_att_scores.append(att_score)
            meta_view_tmp.append(att_res)
        meta_view = torch.stack(meta_view_tmp, dim=2)  # [N, schema_num+1, edge_type_cnt, dim]
        meta_view_att_scores = torch.stack(meta_view_att_scores, dim=1)  # [N, edge_type_cnt, n_head, schema_num+1, schema_num+1]

        # batch * embed
        # out_embed = torch.matmul(meta_view[edgetype[0], metatype, edgetype[1]].unsqueeze(dim=1),
        #                          self.reflect[metatype, edgetype[1]]).squeeze(dim=1)
        out_embed = torch.matmul(torch.mean(meta_view, dim=1)[edgetype[0], edgetype[1]].unsqueeze(dim=1),
                                 self.reflect[edgetype[1]]).squeeze(dim=1)
        if self.att_vis == True:
            return F.normalize(base_embedding + out_embed, dim=1), meta_view_att_scores
        else:
            return F.normalize(base_embedding + out_embed, dim=1), None


class SuperHybridGNN(nn.Module):
    name = "hybrid"

    '''
        schema: "U-I-U, U-A-I-A-U" 
    '''

    def __init__(self, max_users, edge_type_count, schema=None, edge_dim=100, input_dim=200, output_dim=200,
                 random_layer=3):
        super(SuperHybridGNN, self).__init__()
        # Param
        self.alpha = 0.1
        self.schema_types = schema.replace(' ', '').split(",")
        self.embedding_size = input_dim
        self.edge_dim = edge_dim
        self.max_users = max_users
        self.edge_type = edge_type_count
        self.partition_list = self.get_partition_list()
        self.random_partition_list = self.get_random_partition_list(random_layer)
        self.random_sample_layer = random_layer

        # hybrid mixture
        self.node_type_embeddings = nn.Parameter(
            torch.FloatTensor(self.max_users, edge_type_count, edge_dim)
        )

        meta_agg_layers = []
        for idx in range(len(self.schema_types)):
            agg_layers = []
            for spec_layer in range(len(self.schema_types[idx].split("-")) - 1):
                agg = MeanAggregator(input_dim=edge_dim, output_dim=edge_dim // 2)
                agg_layers.append(agg)
            self.agg_layers = nn.Sequential(*agg_layers)
            meta_agg_layers.append(self.agg_layers)
        self.meta_agg_layer = nn.Sequential(*meta_agg_layers)
        self.random_agg_layer = nn.Sequential(*[
            MeanAggregator(input_dim=edge_dim, output_dim=edge_dim // 2),
            MeanAggregator(input_dim=edge_dim, output_dim=edge_dim // 2),
        ])
        self.fc = nn.Linear(input_dim, output_dim, bias=True)

        # Network
        # self.reflect = nn.Parameter(torch.FloatTensor(len(self.schema_types), edge_type_count, edge_dim, output_dim))
        self.reflect = nn.Parameter(torch.FloatTensor(edge_type_count, edge_dim, output_dim))
        self.base_embed = nn.Embedding(self.max_users, input_dim)
        self.rw_embed = nn.Parameter(
            torch.FloatTensor(self.max_users, edge_dim)
        )
        self.type_embed = nn.Parameter(
            torch.FloatTensor(self.max_users, edge_type_count, len(self.schema_types), edge_dim))
        self.slf_meta_attn = MultiHeadAttention(n_head=1, d_model=edge_dim, d_k=edge_dim, d_v=edge_dim)
        self.slf_view_attn = MultiHeadAttention(n_head=1, d_model=edge_dim, d_k=edge_dim, d_v=edge_dim)
        self.param_reset()

    def param_reset(self):
        self.base_embed.weight.data.uniform_(-1.0, 1.0)
        self.reflect.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.type_embed.data.uniform_(-1.0, 1.0)
        self.rw_embed.data.uniform_(-1.0, 1.0)

    def get_random_partition_list(self, random_layers):
        random_partition_list = []
        for ilayer in range(1, random_layers):
            start = 1
            for num in range(2 * ilayer + 1, 0, -2):
                start *= num
            random_partition_list.append(start)
        return random_partition_list

    def get_partition_list(self):
        schema_len_list = []
        for schema in self.schema_types:
            schema_len_list.append(len(schema.split("-")))
        partition_list = []
        for idx in range(len(schema_len_list)):
            tmp = []
            for ilayer in range(1, schema_len_list[idx]):
                start = 1
                for num in range(2 * ilayer + 1, 0, -2):
                    start *= num
                tmp.append(start)
            partition_list.append(tmp)
        return partition_list

    def forward(self, nodeids, edgetype, neighbors, random_neighbors):
        '''
        :param nodeids: B
        :param edgetype: [B, L]
        :param neighbors: [B * T * N]
        :param random_neighbors: [B * N]
        :return:
        '''
        # edge tpye embedding: self.max_users, edge_type_count, len(self.schema_types), edge_dim)
        base_embedding = self.base_embed(nodeids)
        # 不同metapath邻居拆分, 得到不同关系下的同一个metapath
        neighbors = list(torch.split(neighbors, [sum(overall) for overall in self.partition_list], dim=2))
        edgetype = torch.split(edgetype, 1, dim=1)  # 得到每一层的edgetype

        # batch * types * (3 * 5), batch * types * (3 * 5 * 7) ....
        # loop schema
        meta_view = []
        for idx, neigh in enumerate(neighbors):  # 总metapath的数量
            all_feats = []  # (self.max_users, edge_type_count, edge_dim))
            base = self.type_embed[nodeids]  # batch * type * meta * embed
            # batch * type * meta * embed
            layer_neigh = list(neighbors[idx].split(self.partition_list[idx], dim=2))
            # batch * types * (batch * 3), batch * types * (batch * 3 * 5)
            for neigh in layer_neigh:
                all_feats.append(self.type_embed[neigh][:, :, :, :, idx, :])  # batch * type * num * type * embed
                # batch * type * num * type * meta * dim
            spec_embed_list = []
            for typeid in range(base.size(1)):
                # (batch * num) * embed
                type_feats = [feats[:, typeid, :, typeid, :].contiguous().view(-1, self.edge_dim) for feats in
                              all_feats]
                type_feats.insert(0, base[:, typeid, idx, :])
                # print(len(self.meta_agg_layer[idx].children()))
                for agg_layer in self.meta_agg_layer[idx].children():
                    type_feats = [agg_layer(type_feats[k], type_feats[k + 1]) for k in range(len(type_feats) - 1)]
                assert len(type_feats) == 1, "asseration len(all_feats) == 1"
                spec_embed_list.append(type_feats[0])  # batch * embed
            # [] len: types, batch * embed
            spec_embed = torch.stack(spec_embed_list, dim=1)  # batch * type * embed
            meta_view.append(spec_embed)

        # all_feats = []  # (self.max_users, edge_type_count, edge_dim))
        # layer_neigh = list(random_neighbors.split(self.random_partition_list, dim=1))  # B * 3, B * (3*5)
        # for neigh in layer_neigh:
        #     all_feats.append(self.rw_embed[neigh].contiguous().view(-1, self.edge_dim))  # batch * embed
        # all_feats.insert(0, self.rw_embed[nodeids])
        # # print(len(self.meta_agg_layer[idx].children()))
        # for agg_layer in self.random_agg_layer.children():
        #     all_feats = [agg_layer(all_feats[k], all_feats[k + 1]) for k in range(len(all_feats) - 1)]
        # assert len(all_feats) == 1, "asseration len(all_feats) == 1"
        # random_view = all_feats[0].unsqueeze(dim=1)
        #
        # meta_view = [torch.cat([view, random_view], dim=1) for view in meta_view]

        # self attention on type
        meta_view = [self.slf_view_attn(view, view, view)[0] for view in meta_view]
        meta_view = torch.stack(meta_view, dim=1)  # batch * meta * type * embed embeding
        # type embedding
        # self attention on meta; [] len: typeid, batch * meta * embed
        meta_view = [self.slf_meta_attn(meta_view[:, :, typeid, :],
                                        meta_view[:, :, typeid, :],
                                        meta_view[:, :, typeid, :])[0] for typeid in range(meta_view.size(2))]
        meta_view = torch.stack(meta_view, dim=2)  # batch * meta * type * embed

        # batch * embed
        # out_embed = torch.matmul(meta_view[edgetype[0], metatype, edgetype[1]].unsqueeze(dim=1),
        #                          self.reflect[metatype, edgetype[1]]).squeeze(dim=1)
        out_embed = torch.matmul(torch.mean(meta_view, dim=1)[edgetype[0], edgetype[1]].unsqueeze(dim=1),
                                 self.reflect[edgetype[1]]).squeeze(dim=1)
        return F.normalize(base_embedding + out_embed, dim=1)


if __name__ == '__main__':
    g2 = [(1, 2), (1, 3), (1, 4), (2, 3)]
    g3 = [(1, 3), (1, 4), (2, 3)]
    g = {"a": g2, "b": g3}
    nodetype = {1: 'U', 2: 'I', 3: 'A', 4: 'I'}
    schema = "U-I-U, U-A-U"
    # meta_sager = MetaGraphSager(graph=g2, schema=schema.split(",")[0], nodetype=nodetype)
    # a = meta_sager.meta_sampler(torch.tensor([1,2]).long())
    # print(a)
    # n = 2
    # b = [torch.split(t, [t.size(0) // n, t.size(0) // n], dim=0) for t in a]
    # print(b)
    # c = []
    # for i in range(n):
    #     tmp = []
    #     for bb in b:
    #         tmp.append(bb[i])
    #     c.append(tmp)
    #
    # print(c)
    # c = [[bb[i] for bb in b] for i in range(n)]
    # print(c)
    # f_neighs = []
    # all_agg_neighbors = a
    # all_nodes = [1, 2]
    # for flatten in [all_agg_neighbors]:
    #     tensors = flatten
    #     agg_tensors = [torch.split(tensor, [tensor.size(0) // len(all_nodes) for _ in range(len(all_nodes))], dim=0)
    #                    for tensor in tensors]
    #     for idx in range(len(all_nodes)):
    #         neighs = [agg_tensor[idx] for agg_tensor in agg_tensors]
    #         f_neighs.append(neighs)
    # all_agg_neighbors = f_neighs
    # print(all_agg_neighbors)
    print("==========")
    meta = HybridGNN(
        max_users=5, edge_type_count=5, schema=schema, edge_dim=10, input_dim=200, output_dim=200
    )
    nodeids = torch.tensor([3, 1]).long()
    edgetype = torch.tensor([0, 1]).long()
    random_neighbors = torch.tensor([
        [
            [1, 1], [2, 2], [3, 3, ], [0, 0], [4, 4]
        ],
        [
            [1, 2], [2, 1], [3, 4], [4, 3], [0, 0]
        ]
    ]).long()
    neighbors = torch.tensor([
        [[1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],
         [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, ],
         [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, ],
         [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, ],
         [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, ]
         ],
        [[1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],
         [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, ],
         [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, ],
         [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, ],
         [1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, ]
         ]]
    )
    a = torch.tensor([i for i in range(edgetype.size(0))]).long()
    # edgetype = list(zip(*[(idx, item) for idx, item in enumerate(edgetype)]))
    # print(edgetype)
    embed = meta(nodeids, (a, edgetype), neighbors, random_neighbors)
    print(embed.size())

