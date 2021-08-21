import sys
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import Model

sys.path.append("..")
from load_data import load_train_data, load_test_data

import random

parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epoch. Default: 1000")
parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate during optimization. Default: 1e-2")
parser.add_argument("--drop_rate", type=float, default=0.3, help="Drop rate of the Dropout Layer. Default: 0.3")
parser.add_argument("--lambda1", type=float, default=1e-5, help="Lambda1 for balancing loss. Default: 1e-5")
parser.add_argument("--is_train", default=True, action="store_false", help="True to train and False to inference. Default: True")
parser.add_argument("--train_dir", type=str, default="./train", help="Training directory for saving model. Default: ./train")
parser.add_argument("--test_dir", type=str, default="./test/test.npy", help="Testing directory for saving prediction result. Default: ./test/test.npy")
parser.add_argument("--inference_version", type=int, default=0, help="The version for inference. Set 0 to use latest checkpoint. Default: 0")
parser.add_argument("--alpha", type=float, default=1.0, help="Coefficient for variance loss. Default: 1.0")
args = parser.parse_args()


def train_epoch(model, X, y, optimizer):  # Training Process
    model.train()
    optimizer.zero_grad()
    loss, acc = model(torch.from_numpy(X).to(device), torch.from_numpy(y).to(device))
    loss.backward()
    optimizer.step()
    return acc.item(), loss.item()


def valid_epoch(model, X, y):  # Valid Process
    model.eval()
    loss, acc = model(torch.from_numpy(X).to(device), torch.from_numpy(y).to(device))
    return acc.item(), loss.item()


def inference(model, X):  # Test Process
    model.eval()
    pred_ = model(torch.from_numpy(X).to(device))
    return pred_.cpu().data.numpy()


seed = 521

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)  # seed of cpu
    torch.cuda.manual_seed_all(seed)  # seed of all GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True  # default
    torch.backends.cudnn.benchmark = False  # default
    torch.backends.cudnn.deterministic = True  # default: False; if benchmark is True, must be Ture

    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    if args.is_train:
        train_accs, val_accs = [], []
        X_train, y_train, X_val, y_val, X_test1, y_test1, X_test2, y_test2 = load_train_data()
        model = Model(X_train.shape[1], drop_rate=args.drop_rate, l1=args.lambda1, alpha=args.alpha)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        pre_losses = [1e18] * 3
        best_val_acc = 0.0
        for epoch in range(1, args.num_epochs + 1):
            start_time = time.time()
            train_acc, train_loss = train_epoch(model, X_train, y_train, optimizer)

            val_acc, val_loss = valid_epoch(model, X_val, y_val)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                with open(os.path.join(args.train_dir, "checkpoint_{}.pth.tar".format(epoch)), "wb") as fout:
                    torch.save(model, fout)
                with open(os.path.join(args.train_dir, "checkpoint_0.pth.tar"), "wb") as fout:
                    torch.save(model, fout)

            epoch_time = time.time() - start_time
            print("Epoch " + str(epoch) + " of " + str(args.num_epochs) + " took " + str(epoch_time) + "s")
            print("  learning rate:                 " + str(optimizer.param_groups[0]["lr"]))
            print("  training loss:                 " + str(train_loss))
            print("  training accuracy:             " + str(train_acc))
            print("  validation loss:               " + str(val_loss))
            print("  validation accuracy:           " + str(val_acc))
            print("  best epoch:                    " + str(best_epoch))
            print("  best validation accuracy:      " + str(best_val_acc))
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            if train_loss > max(pre_losses):
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.995
            pre_losses = pre_losses[1:] + [train_loss]
            test1_acc, _ = valid_epoch(model, X_test1, y_test1)
            test2_acc, _ = valid_epoch(model, X_test2, y_test2)
            print(test1_acc, test2_acc)
    else:
        print("begin testing")
        model_path = os.path.join(args.train_dir, "checkpoint_%d.pth.tar" % args.inference_version)
        if os.path.exists(model_path):
            model = torch.load(model_path)
        X_train, y_train, X_val, y_val = load_train_data()
        f = model.get_middle(torch.from_numpy(X_train).to(device))
        np.save("features.npy", f.cpu().data.numpy())
        X_test = load_test_data()

        result = inference(model, X_test)
        np.save(args.test_dir, result)
