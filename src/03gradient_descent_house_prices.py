"""使用批量梯度下降拟合一元线性房价模型。"""

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def create_house_data(random_state=42):
    """生成面积与房价的模拟数据。"""
    rng = np.random.RandomState(random_state)
    areas_m2 = rng.uniform(90, 150, 100)
    areas_100m2 = areas_m2 / 100
    prices = 250 * areas_100m2 + 30 + rng.randn(100) * 15
    return areas_m2, areas_100m2, prices


def gradient_descent(X, y, learning_rate=0.01, epochs=100_000):
    """最小化均方误差，并记录每次更新后的参数与损失。"""
    weight, bias = 0.0, 0.0
    sample_count = len(X)
    history = {
        "weight": [weight],
        "bias": [bias],
        "loss": [np.mean((weight * X + bias - y) ** 2)],
    }
    print_interval = max(1, epochs // 100)

    for epoch in range(epochs):
        prediction = weight * X + bias
        error = prediction - y
        weight_gradient = (2 / sample_count) * np.sum(error * X)
        bias_gradient = (2 / sample_count) * np.sum(error)
        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

        current_loss = np.mean((weight * X + bias - y) ** 2)
        history["weight"].append(weight)
        history["bias"].append(bias)
        history["loss"].append(current_loss)

        if epoch % print_interval == 0 or epoch == epochs - 1:
            print(
                f"迭代 {epoch + 1:>6}: weight={weight:.4f}, "
                f"bias={bias:.4f}, loss={current_loss:.4f}"
            )
    return history


def plot_training(areas_m2, prices, history, skip=100):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    final_weight = history["weight"][-1]
    final_bias = history["bias"][-1]

    axes[0].scatter(areas_m2, prices, color="steelblue", s=60, alpha=0.7, label="房屋数据")
    x_line = np.linspace(85, 155, 100)
    y_line = final_weight / 100 * x_line + final_bias
    axes[0].plot(
        x_line,
        y_line,
        color="coral",
        linewidth=2,
        label=f"拟合: y={final_weight / 100:.3f}x+{final_bias:.1f}",
    )
    axes[0].set_xlabel("面积（平方米）")
    axes[0].set_ylabel("房价（万）")
    axes[0].set_title("房价预测模型")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(250, color="steelblue", linestyle="--", alpha=0.5)
    axes[1].plot(history["weight"][skip:], color="steelblue", label="weight（目标 250）")
    axes[1].axhline(30, color="darkorange", linestyle="--", alpha=0.5)
    axes[1].plot(history["bias"][skip:], color="darkorange", label="bias（目标 30）")
    axes[1].set_xlabel(f"迭代次数（跳过前 {skip} 次）")
    axes[1].set_ylabel("参数值")
    axes[1].set_title("参数收敛过程")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["loss"][skip:], color="orangered")
    axes[2].set_xlabel(f"迭代次数（跳过前 {skip} 次）")
    axes[2].set_ylabel("MSE")
    axes[2].set_title("损失函数下降")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()


def main():
    areas_m2, areas, prices = create_house_data()
    print(f"房屋数量: {len(areas)} 套")
    print(f"面积范围: {areas_m2.min():.1f}–{areas_m2.max():.1f} 平方米")
    print(f"价格范围: {prices.min():.1f}–{prices.max():.1f} 万")
    print("真实规律: 房价 = 2.5 × 面积（平方米）+ 30")

    history = gradient_descent(areas, prices)
    print(
        f"\n最终模型: 房价 = {history['weight'][-1] / 100:.3f} "
        f"× 面积（平方米）+ {history['bias'][-1]:.2f}"
    )
    plot_training(areas_m2, prices, history)
    plt.show()


if __name__ == "__main__":
    main()
