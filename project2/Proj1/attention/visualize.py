import numpy as np
from skimage.feature import hog
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from load_data import load_train_data


def visualize():
    X_train, y_train, _, _ = load_train_data()
    features = np.load("features.npy")
    transformed = TSNE(n_components=2, init="pca", random_state=0).fit_transform(np.vstack((X_train, features)))
    transformed_x = transformed[: len(y_train)]
    transformed_f = transformed[len(y_train) :]
    print("ok")
    plt.figure(figsize=(6, 5), dpi=200)
    cnt = [0 for _ in range(10)]
    for i in range(len(y_train)):
        v = int(y_train[i])
        if cnt[v] >= 100:
            continue
        cnt[v] += 1
        plt.scatter(transformed_x[i][0], transformed_x[i][1], s=5, color="w", edgecolor=plt.cm.Set1(v / 10.0), marker="o", linewidths=0.5)
        plt.scatter(transformed_f[i][0], transformed_f[i][1], s=5, color=plt.cm.Set1(v / 10.0), marker="x", linewidths=0.5)
    patch_lst = []
    for i in range(10):
        patch_lst.append(mpatches.Patch(color=plt.cm.Set1(i / 10.0), label=str(i)))
    plt.legend(handles=patch_lst, fontsize=6)
    plt.savefig("cluster.png")


visualize()