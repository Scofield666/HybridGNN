import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import conf


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size=200):
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
        ).view(n, self.num_sampled).to(conf.device)
        noise = torch.neg(self.context_embedding(negs)).to(conf.device)
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, embed.unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n
