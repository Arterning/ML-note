"""使用 scikit-learn 演示肘部法和 K-Means++ 聚类。"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def demonstrate_elbow_method():
    """计算不同 k 对应的 SSE，并绘制肘部法曲线。"""
    X, _ = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=0.6,
        random_state=0,
    )

    k_range = range(1, 11)
    sse_list = []
    for k in k_range:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=42,
        )
        model.fit(X)
        sse_list.append(model.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(k_range, sse_list, "bo-", linewidth=2, markersize=4)
    plt.xlabel("簇数 k")
    plt.ylabel("SSE（簇内误差平方和）")
    plt.title("肘部法则选择 k")
    plt.xticks(list(k_range))
    plt.grid(True, alpha=0.3)
    plt.annotate(
        "肘部",
        xy=(4, sse_list[3]),
        xytext=(6, sse_list[3] + 100),
        fontsize=12,
        color="red",
        arrowprops={"arrowstyle": "->", "color": "red"},
    )
    plt.tight_layout()


def demonstrate_kmeans_plus_plus():
    """使用 K-Means++ 完成聚类，并对比聚类前后的样本。"""
    X, _ = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=0.8,
        random_state=42,
    )

    model = KMeans(
        n_clusters=4,
        init="k-means++",
        n_init=10,
        random_state=42,
    )
    predicted_labels = model.fit_predict(X)
    centroids = model.cluster_centers_

    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(X[:, 0], X[:, 1], c="gray", s=20, alpha=0.6)
    axes[0].set_title("聚类前")
    axes[0].set_xlabel("特征 1")
    axes[0].set_ylabel("特征 2")

    axes[1].scatter(
        X[:, 0], X[:, 1], c=predicted_labels, s=20, cmap="viridis", alpha=0.6
    )
    axes[1].scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        s=200,
        marker="x",
        linewidths=1.5,
        label="质心",
    )
    axes[1].set_title("K-Means 聚类结果（k=4）")
    axes[1].set_xlabel("特征 1")
    axes[1].set_ylabel("特征 2")
    axes[1].legend()
    plt.tight_layout()

    print(f"SSE: {model.inertia_:.2f}")
    print(f"迭代次数: {model.n_iter_}")


def main():
    demonstrate_elbow_method()
    demonstrate_kmeans_plus_plus()
    plt.show()


if __name__ == "__main__":
    main()
