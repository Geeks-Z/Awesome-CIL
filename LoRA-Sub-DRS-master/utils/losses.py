from __future__ import print_function

import torch
import torch.nn as nn
import sys
import numpy as np
import torch.nn.functional as F

class AugmentedTripletLoss(nn.Module):
    def __init__(self, margin=1.0, norm=2):
        super(AugmentedTripletLoss, self).__init__()
        self.margin = margin
        self.norm = norm
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, center):
        # device = (torch.device('cuda')
        #           if inputs.is_cuda
        #           else torch.device('cpu'))
        device = (inputs.device
                  if inputs.is_cuda
                  else torch.device('cpu'))
        n = inputs.size(0)  # batch_size

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        num_proto = len(center)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            if dist[i][mask[i] == 0].numel() == 0:
                dist_an.append((dist[i][mask[i]].max()+self.margin).unsqueeze(0))
            else:
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        if num_proto > 0:
            center = torch.from_numpy(center / np.linalg.norm(center, axis=1)[:, None]).to(device)
            for i in range(n):
                for j in range(num_proto):
                    distp = torch.norm(inputs[i].unsqueeze(0) - center[j], self.norm).clamp(min=1e-12)
                    dist_an[i] = min(dist_an[i].squeeze(0), distp).unsqueeze(0)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


# def compute_distance(x, centers):
#     features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
#     centers_square = torch.sum(torch.pow(centers, 2), 0, keepdim=True)
#     features_into_centers = 2 * torch.matmul(x, (centers))
#     dist = features_square + centers_square - features_into_centers
#     dist = dist / float(x.shape[1])
#     return dist





