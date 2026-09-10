"""比较无正则化、Ridge 和 Lasso 对高次多项式过拟合的影响。"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def create_data(random_state=42):
    rng = np.random.default_rng(random_state)
    X = np.sort(rng.uniform(-3, 3, 30)).reshape(-1, 1)
    y = X.ravel() ** 2 + rng.normal(0, 1.5, 30)
    return X, y


def create_models(degree=15):
    """先扩展多项式，再标准化，最后进行回归。"""
    return {
        "普通线性回归\n（无正则化）": make_pipeline(
            PolynomialFeatures(degree),
            StandardScaler(),
            LinearRegression(),
        ),
        "Ridge（L2）\n（整体压缩权重）": make_pipeline(
            PolynomialFeatures(degree),
            StandardScaler(),
            Ridge(alpha=5.0),
        ),
        "Lasso（L1）\n（产生稀疏权重）": make_pipeline(
            PolynomialFeatures(degree),
            StandardScaler(),
            Lasso(alpha=0.2, max_iter=10_000),
        ),
    }


def main():
    X_train, y_train = create_data()
    X_plot = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
    models = create_models()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for column, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        regressor = model.steps[-1][1]
        coefficients = regressor.coef_
        nonzero_count = np.count_nonzero(np.abs(coefficients) > 1e-5)

        curve_axis = axes[0, column]
        curve_axis.scatter(X_train[:, 0], y_train, color="black", s=30, label="训练数据")
        curve_axis.plot(
            X_plot[:, 0],
            X_plot.ravel() ** 2,
            "--",
            color="gray",
            label=r"真实关系 $y=x^2$",
        )
        curve_axis.plot(
            X_plot[:, 0],
            model.predict(X_plot),
            color="red",
            linewidth=2,
            label="模型预测",
        )
        curve_axis.set_title(name)
        curve_axis.set_ylim(-5, 15)
        curve_axis.grid(True, alpha=0.3)
        curve_axis.legend()

        coefficient_axis = axes[1, column]
        bars = coefficient_axis.bar(range(len(coefficients)), coefficients, color="steelblue")
        for bar, coefficient in zip(bars, coefficients):
            if abs(coefficient) <= 1e-5:
                bar.set_color("lightgray")
        coefficient_axis.axhline(0, color="black", linewidth=0.8)
        coefficient_axis.set_title(
            f"模型系数分布\n非零系数: {nonzero_count}/{len(coefficients)}"
        )
        coefficient_axis.set_xlabel("多项式特征序号")
        coefficient_axis.set_ylabel("系数值")
        coefficient_axis.grid(axis="y", alpha=0.3)
        print(f"{name.replace(chr(10), ' ')}: 非零系数 {nonzero_count}/{len(coefficients)}")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
