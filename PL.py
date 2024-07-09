import torch
import torch.nn as nn
import torch.nn.functional as F
from Dist import Dist

class PL(nn.CrossEntropyLoss):
    def __init__(self, args):
        super(PL, self).__init__()
        self.args = args
        self.use_gpu = True
        self.weight_pl = float(args.weight_pl)
        self.temp = args.temp
        self.Dist = Dist(num_classes=args.n_known_class, num_centers=args.num_centers, feat_dim=args.feature_size)
        self.pos_points = self.Dist.pos_centers
        self.neg_points = self.Dist.neg_centers

    def forward(self, x, labels=None):
        dist_dot_pos = self.Dist(x, center=self.pos_points)
        dist_dot_neg = self.Dist(x, center=self.neg_points)
        logits = dist_dot_pos - dist_dot_neg  # (batch_size, class_num)
        logits = logits / self.temp

        if labels is None:
            return logits, 0

        true_logits = (torch.exp(logits) * labels).sum(dim=0)
        false_logits = torch.exp(logits).sum(dim=0)

        loss = - torch.log(true_logits / false_logits + 1e-3).mean()

        # loss = F.cross_entropy(logits, labels)

        return logits, loss
