"""从零实现 K-Means，并可视化每一轮的分配和质心更新。"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def create_data():
    """生成 notebook 中使用的二维聚类样本。"""
    X, y_true = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=0.6,
        random_state=0,
    )
    return X, y_true


def initialize_centroids(X, k, random_state=42):
    """从样本中随机选择 k 个点作为初始质心。"""
    rng = np.random.default_rng(random_state)
    initial_indices = rng.choice(X.shape[0], k, replace=False)
    return X[initial_indices].copy()


def assign_clusters(X, centroids):
    """把每个样本分配给距离最近的质心。"""
    distances = cdist(X, centroids)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, k, old_centroids):
    """以每个簇的样本均值更新质心，并处理可能出现的空簇。"""
    new_centroids = []
    for cluster_id in range(k):
        cluster_points = X[labels == cluster_id]
        if len(cluster_points) == 0:
            new_centroids.append(old_centroids[cluster_id])
        else:
            new_centroids.append(cluster_points.mean(axis=0))
    return np.asarray(new_centroids)


def fit_kmeans(X, k=4, max_iters=15, tolerance=1e-8):
    """执行 K-Means，返回最终结果以及每轮迭代状态。"""
    centroids = initialize_centroids(X, k)
    history = []

    for iteration in range(1, max_iters + 1):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k, centroids)
        history.append((centroids.copy(), new_centroids.copy(), labels.copy()))

        if np.allclose(centroids, new_centroids, atol=tolerance, rtol=0):
            print(f"算法在第 {iteration} 次迭代后收敛，质心不再移动。")
            return new_centroids, labels, history

        centroids = new_centroids

    labels = assign_clusters(X, centroids)
    print(f"算法达到最大迭代次数 {max_iters}。")
    return centroids, labels, history


def plot_initial_data(X):
    """展示生成的原始样本。"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], s=30, color="gray")
    plt.title("初始化数据")
    plt.tight_layout()


def plot_history(X, history):
    """并排展示每轮的标签分配和质心更新。"""
    row_count = len(history)
    fig, axes = plt.subplots(row_count, 2, figsize=(14, 5 * row_count), squeeze=False)

    for row, (old_centroids, new_centroids, labels) in enumerate(history):
        assignment_ax, update_ax = axes[row]

        assignment_ax.scatter(
            X[:, 0], X[:, 1], c=labels, s=30, cmap="viridis", alpha=0.6
        )
        assignment_ax.scatter(
            old_centroids[:, 0],
            old_centroids[:, 1],
            c="red",
            s=200,
            marker="x",
            linewidths=2,
        )
        assignment_ax.set_title(f"迭代 {row + 1}：分配点到最近质心")

        update_ax.scatter(
            X[:, 0], X[:, 1], c=labels, s=30, cmap="viridis", alpha=0.6
        )
        update_ax.scatter(
            new_centroids[:, 0],
            new_centroids[:, 1],
            c="red",
            s=200,
            marker="x",
            linewidths=2,
        )
        update_ax.set_title(f"迭代 {row + 1}：更新质心到簇均值")

    fig.suptitle("K-Means 逐轮迭代", fontsize=14, fontweight="bold")
    fig.tight_layout()


def main():
    X, _ = create_data()
    print(f"样本矩阵形状: {X.shape}")

    plot_initial_data(X)
    centroids, labels, history = fit_kmeans(X, k=4, max_iters=15)
    print("最终质心：")
    print(np.round(centroids, 3))
    print(f"迭代轮数: {len(history)}")

    plot_history(X, history)
    plt.show()


if __name__ == "__main__":
    main()
