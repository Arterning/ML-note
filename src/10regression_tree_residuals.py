"""用极简数据解释回归树的 MSE 分裂和 GBDT 残差拟合。"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

X_SIMPLE = np.arange(1, 9).reshape(-1, 1)
Y_SIMPLE = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0])


def calculate_split_mse(X, y, threshold):
    """计算指定阈值分裂后的加权 MSE。"""
    left = y[X.ravel() <= threshold]
    right = y[X.ravel() > threshold]
    left_mse = np.mean((left - left.mean()) ** 2) if len(left) else 0
    right_mse = np.mean((right - right.mean()) ** 2) if len(right) else 0
    return (len(left) * left_mse + len(right) * right_mse) / len(y)


def show_split_candidates(X, y):
    candidates = ((X[:-1] + X[1:]) / 2).ravel()
    original_mse = np.mean((y - y.mean()) ** 2)
    print(f"{'分裂点':>10} {'加权 MSE':>12} {'MSE 下降':>12}")
    print("-" * 38)
    for threshold in candidates:
        split_mse = calculate_split_mse(X, y, threshold)
        print(f"X <= {threshold:>3.1f} {split_mse:>12.4f} {original_mse - split_mse:>12.4f}")
    print(f"原始 MSE（全部预测为均值 {y.mean():.2f}）: {original_mse:.4f}")


def plot_original_tree(X, y):
    tree_model = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, y)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].scatter(X[:, 0], y, s=80, color="steelblue")
    axes[0].plot(X[:, 0], y, "--", color="gray", alpha=0.5)
    axes[0].set_title('原始数据：一个对称的“山峰”')
    axes[0].grid(alpha=0.3)
    plot_tree(tree_model, feature_names=["X"], filled=True, rounded=True, ax=axes[1])
    axes[1].set_title("回归决策树（max_depth=2）")
    fig.tight_layout()
    print("\n单棵回归树预测:", tree_model.predict(X))


def plot_two_boosting_rounds(X, y):
    initial_prediction = np.full_like(y, y.mean())
    first_residual = y - initial_prediction
    first_tree = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, first_residual)
    first_prediction = initial_prediction + first_tree.predict(X)

    second_residual = y - first_prediction
    second_tree = DecisionTreeRegressor(max_depth=2, random_state=43).fit(X, second_residual)
    second_prediction = first_prediction + second_tree.predict(X)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes[0, 0].scatter(X[:, 0], y, s=60, color="steelblue")
    axes[0, 0].axhline(initial_prediction[0], color="gray", linestyle="--")
    axes[0, 0].set_title("第 0 步：用均值初始化")
    axes[0, 1].scatter(X[:, 0], first_residual, s=60, color="darkorange")
    axes[0, 1].axhline(0, color="gray", linestyle="--")
    axes[0, 1].set_title("第 1 轮要拟合的残差")
    plot_tree(first_tree, feature_names=["X"], filled=True, ax=axes[0, 2])
    axes[0, 2].set_title("第一棵树")

    axes[1, 0].scatter(X[:, 0], y, s=60, color="steelblue")
    axes[1, 0].plot(X[:, 0], first_prediction, "o-", color="tomato")
    axes[1, 0].set_title("第 1 轮后的预测")
    axes[1, 1].scatter(X[:, 0], second_residual, s=60, color="darkorange")
    axes[1, 1].axhline(0, color="gray", linestyle="--")
    axes[1, 1].set_title("第 2 轮要拟合的新残差")
    plot_tree(second_tree, feature_names=["X"], filled=True, ax=axes[1, 2])
    axes[1, 2].set_title("第二棵树")
    fig.tight_layout()

    print("\n预测误差变化：")
    print(f"初始化 MSE: {np.mean((y - initial_prediction) ** 2):.4f}")
    print(f"第 1 轮 MSE: {np.mean((y - first_prediction) ** 2):.4f}")
    print(f"第 2 轮 MSE: {np.mean((y - second_prediction) ** 2):.4f}")


def main():
    show_split_candidates(X_SIMPLE, Y_SIMPLE)
    plot_original_tree(X_SIMPLE, Y_SIMPLE)
    plot_two_boosting_rounds(X_SIMPLE, Y_SIMPLE)
    plt.show()


if __name__ == "__main__":
    main()
