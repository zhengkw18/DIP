import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_feature, drop_rate, l1):
        super(Model, self).__init__()

        self.fc0 = Parameter(torch.randn(num_feature, 512).to(torch.double) / num_feature)
        self.fc1 = Parameter(torch.randn(num_feature, 512).to(torch.double) / num_feature)

        self.g = Parameter(torch.randn(num_feature, 1).to(torch.double) / num_feature)

        self.w = Parameter(torch.randn(num_feature, 256).to(torch.double) / num_feature)
        self.seq = nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate))
        self.w1 = Parameter(torch.randn(256, 10).to(torch.double) / 256)
        self.loss = nn.CrossEntropyLoss()

        self.l1 = l1

    def forward(self, x, y=None, Lg=False):
        # x.shape == (num_sample, num_feature)
        n, p = x.shape
        x1 = torch.mm(x, self.fc0)
        x2 = F.relu(torch.mm(x, self.fc1))
        x3 = (1 + F.sigmoid(x1)) * x2
        x4 = x + x3
        logits = torch.mm(self.seq(torch.mm(x4, self.w)), self.w1)

        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred

        M = self.g.expand(-1, p) * (1 - torch.eye(p, device=x.device))
        U = x4.mm(M)
        if Lg:
            loss = 10 * (x4 - U).square().mean()
        else:
            loss = self.loss(logits, y.long()) + self.l1 * (torch.abs(self.w1).sum() + torch.abs(self.w).sum() + torch.abs(self.fc0).sum() + torch.abs(self.fc1).sum())
            X = U.T.mm(x4) * (1 - torch.eye(p, device=x.device))
            loss += 10 * X.square().mean() / n
        correct_pred = pred.int() == y.int()
        acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

        return loss, acc
