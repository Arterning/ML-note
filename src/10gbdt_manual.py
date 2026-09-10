"""用浅层回归树手动实现简化版 GBDT。"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def create_data(random_state=42):
    rng = np.random.default_rng(random_state)
    X = np.sort(rng.random((200, 1)) * 10, axis=0)
    y = np.sin(X).ravel() + rng.normal(0, 0.2, size=200)
    return X, y


def fit_manual_gbdt(X, y, n_trees=9, learning_rate=0.3):
    """逐轮拟合当前残差，返回树、预测轨迹和残差轨迹。"""
    prediction = np.full_like(y, y.mean())
    trees = []
    prediction_history = []
    residual_history = []

    for round_index in range(n_trees):
        residual = y - prediction
        estimator = DecisionTreeRegressor(max_depth=2, random_state=round_index)
        estimator.fit(X, residual)
        prediction += learning_rate * estimator.predict(X)

        trees.append(estimator)
        residual_history.append(residual.copy())
        prediction_history.append(prediction.copy())

    return trees, prediction_history, residual_history


def plot_prediction_process(X, y, prediction_history):
    fig, axes = plt.subplots(3, 3, figsize=(16, 9))
    for index, (axis, prediction) in enumerate(zip(axes.flat, prediction_history)):
        mse = np.mean((y - prediction) ** 2)
        axis.scatter(X[:, 0], y, s=10, alpha=0.4, color="steelblue", label="真实数据")
        axis.plot(X[:, 0], prediction, color="tomato", linewidth=2, label="GBDT 预测")
        axis.set_title(f"第 {index + 1} 棵树后 | MSE={mse:.4f}")
        axis.legend(fontsize=9)
    fig.suptitle("手动 GBDT：逐步逼近真实函数", fontsize=14, fontweight="bold")
    fig.tight_layout()


def plot_residual_process(X, residual_history):
    fig, axes = plt.subplots(3, 3, figsize=(16, 9))
    for index, (axis, residual) in enumerate(zip(axes.flat, residual_history)):
        axis.scatter(X[:, 0], residual, s=10, alpha=0.5, color="darkorange")
        axis.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        axis.set_title(f"第 {index + 1} 轮拟合前的残差")
        axis.set_ylim(-1.5, 1.5)
    fig.suptitle("残差逐轮缩小的过程", fontsize=14, fontweight="bold")
    fig.tight_layout()


def main():
    X, y = create_data()
    trees, predictions, residuals = fit_manual_gbdt(X, y)
    print(f"训练的回归树数量: {len(trees)}")
    print(f"初始 MSE: {np.mean((y - y.mean()) ** 2):.4f}")
    print(f"最终 MSE: {np.mean((y - predictions[-1]) ** 2):.4f}")
    plot_prediction_process(X, y, predictions)
    plot_residual_process(X, residuals)
    plt.show()


if __name__ == "__main__":
    main()
