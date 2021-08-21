import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import numpy as np


class Model(nn.Module):
    def __init__(self, num_sample, num_feature, l1=1e-2, l2=0, l3=0.1, l4=100):
        super(Model, self).__init__()

        self.b1 = Parameter(torch.randn(num_feature, 256).to(torch.double) / num_feature)
        self.seq = nn.Sequential(nn.ReLU(), nn.Dropout(p=0.3))
        self.b2 = Parameter(torch.randn(256, 10).to(torch.double) / 256)
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.w = Parameter(torch.randn(num_sample, 1).to(torch.double) / num_sample + 1.0 / math.sqrt(num_sample))
        self.l1, self.l2, self.l3, self.l4 = l1, l2, l3, l4

    def forward(self, x, y=None):
        # x.shape == (num_sample, num_feature)
        n, p = x.shape
        logits = torch.mm(self.seq(torch.mm(x, self.b1)), self.b2)

        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        weight = self.w * self.w
        loss = (self.loss(logits, y.long()) * weight).sum()
        if self.training:
            col = x.T.mm(weight)
            balancing_m = (x.T.mm(torch.diag(weight.reshape(-1))).mm(x) - col.mm(col.T)) * (1 - torch.eye(p, device=x.device))
            loss += self.l1 * (self.b1.abs().sum() + self.b2.abs().sum())
            loss += self.l2 * (balancing_m * balancing_m).sum()
            loss += self.l3 * (weight * weight).sum()
            loss += self.l4 * (weight.sum() - 1) ** 2
        correct_pred = pred.int() == y.int()
        acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

        return loss, acc
