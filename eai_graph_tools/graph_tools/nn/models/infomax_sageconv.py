import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.inits import uniform


# Graphsage-GCN implementation (inductive GCN)
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim, normalize=True, bias=True)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, normalize=True, bias=True)
        self.prelu1 = nn.PReLU(hidden_dim)
        self.prelu2 = nn.PReLU(hidden_dim)

    def forward(self, x, edge_index, node_cnt, corrupt=False):
        if corrupt:
            perm = torch.randperm(node_cnt)
            x = x[perm]

        x = self.conv1(x, edge_index)
        x = self.prelu1(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        x = torch.matmul(x, torch.matmul(self.weight, summary))
        return x


class Infomax(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Infomax, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.discriminator = Discriminator(hidden_dim)
        self.loss = nn.BCEWithLogitsLoss()
        self.node_losses = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, edge_index, node_cnt):
        positive = self.encoder(x, edge_index, node_cnt, corrupt=False)
        negative = self.encoder(x, edge_index, node_cnt, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2

    def nodes_distances(self, noi_ref_embeddings, x, edge_index, node_cnt):
        positive = self.encoder(x, edge_index, node_cnt, corrupt=False)

        node_summary = torch.sigmoid(noi_ref_embeddings.mean(dim=0))
        positive = self.discriminator(positive, node_summary)

        node_losses = self.node_losses(positive, torch.ones_like(positive))
        return node_losses.detach().numpy()

    def nodes_distances_mean_over_interval(self, x, edge_index, node_cnt):
        positive = self.encoder(x, edge_index, node_cnt, corrupt=False)

        summary = torch.sigmoid(positive.mean(dim=0))
        positive_mean = self.discriminator(positive, summary)
        loss_mean = self.loss(positive_mean, torch.ones_like(positive_mean))

        node_summary = torch.sigmoid(positive.mean(dim=0))
        positive = self.discriminator(positive, node_summary)

        node_losses = self.node_losses(positive, torch.ones_like(positive))
        return loss_mean.detach().numpy(), node_losses.detach().numpy()


def train_infomax(infomax_model,
                  infomax_optimizer,
                  data,
                  epoch):
    infomax_model.train()

    if epoch == 50:
        for param_group in infomax_optimizer.param_groups:
            param_group['lr'] = 0.0005

    infomax_optimizer.zero_grad()
    loss = infomax_model(data.x, data.edge_index, data.num_nodes)
    loss.backward()
    infomax_optimizer.step()
    return loss.item()
