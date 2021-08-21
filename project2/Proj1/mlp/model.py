import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


class Model(nn.Module):
    def __init__(self, num_feature, drop_rate, l1):
        super(Model, self).__init__()

        self.w = Parameter(torch.randn(num_feature, 256).to(torch.double) / num_feature)
        self.seq = nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate))
        self.w1 = Parameter(torch.randn(256, 10).to(torch.double) / 256)
        self.loss = nn.CrossEntropyLoss()

        self.l1 = l1

    def forward(self, x, y=None):
        # x.shape == (num_sample, num_feature)
        logits = torch.mm(self.seq(torch.mm(x, self.w)), self.w1)

        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        loss = self.loss(logits, y.long()) + self.l1 * (torch.abs(self.w1).sum() + torch.abs(self.w).sum())
        correct_pred = pred.int() == y.int()
        acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

        return loss, acc
