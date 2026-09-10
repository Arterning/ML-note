"""通过拟合曲线和学习曲线诊断欠拟合与过拟合。"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def create_data(random_state=42):
    """生成带高斯噪声的二次函数样本。"""
    rng = np.random.default_rng(random_state)
    X = np.sort(rng.uniform(-3, 3, 100)).reshape(-1, 1)
    y = X.ravel() ** 2 + rng.normal(0, 1, 100)
    return X, y


def create_models():
    """创建欠拟合、合理拟合和过拟合三个模型。"""
    return {
        "欠拟合（degree=1）": make_pipeline(
            PolynomialFeatures(1), LinearRegression()
        ),
        "良好拟合（degree=2）": make_pipeline(
            PolynomialFeatures(2), LinearRegression()
        ),
        "过拟合（degree=15）": make_pipeline(
            PolynomialFeatures(15), LinearRegression()
        ),
    }


def plot_fitting_results(X, y, models):
    """直观比较不同复杂度模型的预测曲线。"""
    X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for axis, (title, model) in zip(axes, models.items()):
        model.fit(X, y)
        axis.scatter(X[:, 0], y, color="steelblue", s=30, alpha=0.6, label="训练数据")
        axis.plot(
            X_plot[:, 0],
            X_plot.ravel() ** 2,
            "--",
            color="gray",
            linewidth=2,
            label=r"真实函数 $y=x^2$",
        )
        axis.plot(
            X_plot[:, 0],
            model.predict(X_plot),
            color="tomato",
            linewidth=2,
            label="模型预测",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_xlim(-3.5, 3.5)
        axis.set_ylim(-1, 12)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()


def plot_learning_curves(X, y, models):
    """绘制训练 MSE 与交叉验证 MSE 随样本量的变化。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for axis, (title, model) in zip(axes, models.items()):
        train_sizes, train_scores, validation_scores = learning_curve(
            model,
            X,
            y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            scoring="neg_mean_squared_error",
        )
        train_mse = -train_scores.mean(axis=1)
        validation_mse = -validation_scores.mean(axis=1)

        axis.plot(train_sizes, train_mse, "o-", color="steelblue", label="训练误差")
        axis.plot(
            train_sizes,
            validation_mse,
            "o-",
            color="tomato",
            label="验证误差",
        )
        axis.set_title(title)
        axis.set_xlabel("训练样本数")
        axis.set_ylabel("MSE（对数坐标）")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        axis.legend()
        print(
            f"{title}: 最终训练 MSE={train_mse[-1]:.4f}, "
            f"验证 MSE={validation_mse[-1]:.4f}"
        )
    fig.tight_layout()


def main():
    X, y = create_data()
    models = create_models()
    plot_fitting_results(X, y, models)
    plot_learning_curves(X, y, models)
    plt.show()


if __name__ == "__main__":
    main()
