import torch
import torch.nn as nn


class Dist(nn.Module):
    def __init__(self, num_classes=10, num_centers=1, feat_dim=2):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_centers = num_centers

        self.pos_centers = nn.Parameter(0.1 * torch.randn(num_classes * num_centers, self.feat_dim))
        self.neg_centers = nn.Parameter(0.1 * torch.randn(num_classes * num_centers, self.feat_dim))

    def forward(self, features, center):
        dist = features.matmul(center.t())
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2)

        return dist
