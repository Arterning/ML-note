"""从零实现 Adam，并用于拟合一元线性房价模型。"""

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def create_house_data(random_state=42):
    rng = np.random.RandomState(random_state)
    areas_m2 = rng.uniform(90, 150, 100)
    areas = areas_m2 / 100
    prices = 250 * areas + 30 + rng.randn(100) * 15
    return areas_m2, areas, prices


def adam_optimizer(
    X,
    y,
    learning_rate=0.1,
    epochs=50_000,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    """使用 Adam 最小化一元线性模型的均方误差。"""
    weight, bias = 0.0, 0.0
    weight_first_moment = weight_second_moment = 0.0
    bias_first_moment = bias_second_moment = 0.0
    sample_count = len(X)
    history = {
        "weight": [weight],
        "bias": [bias],
        "loss": [np.mean((weight * X + bias - y) ** 2)],
    }
    print_interval = max(1, epochs // 100)

    for epoch in range(epochs):
        time_step = epoch + 1
        error = weight * X + bias - y
        weight_gradient = (2 / sample_count) * np.sum(error * X)
        bias_gradient = (2 / sample_count) * np.sum(error)

        weight_first_moment = beta1 * weight_first_moment + (1 - beta1) * weight_gradient
        bias_first_moment = beta1 * bias_first_moment + (1 - beta1) * bias_gradient
        weight_second_moment = (
            beta2 * weight_second_moment + (1 - beta2) * weight_gradient**2
        )
        bias_second_moment = beta2 * bias_second_moment + (1 - beta2) * bias_gradient**2

        corrected_weight_first = weight_first_moment / (1 - beta1**time_step)
        corrected_bias_first = bias_first_moment / (1 - beta1**time_step)
        corrected_weight_second = weight_second_moment / (1 - beta2**time_step)
        corrected_bias_second = bias_second_moment / (1 - beta2**time_step)

        weight -= learning_rate * corrected_weight_first / (
            np.sqrt(corrected_weight_second) + epsilon
        )
        bias -= learning_rate * corrected_bias_first / (
            np.sqrt(corrected_bias_second) + epsilon
        )
        loss = np.mean((weight * X + bias - y) ** 2)
        history["weight"].append(weight)
        history["bias"].append(bias)
        history["loss"].append(loss)

        if epoch % print_interval == 0 or epoch == epochs - 1:
            print(
                f"迭代 {epoch + 1:>5}: weight={weight:.4f}, "
                f"bias={bias:.4f}, loss={loss:.4f}"
            )
    return history


def plot_training(areas_m2, prices, history, skip=100):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    weight = history["weight"][-1]
    bias = history["bias"][-1]
    x_line = np.linspace(85, 155, 100)

    axes[0].scatter(areas_m2, prices, color="steelblue", s=60, alpha=0.7, label="房屋数据")
    axes[0].plot(
        x_line,
        weight / 100 * x_line + bias,
        color="coral",
        linewidth=2,
        label=f"拟合: y={weight / 100:.3f}x+{bias:.1f}",
    )
    axes[0].set_xlabel("面积（平方米）")
    axes[0].set_ylabel("房价（万）")
    axes[0].set_title("Adam 房价预测模型")
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
    history = adam_optimizer(areas, prices)
    print(
        f"\n最终模型: 房价 = {history['weight'][-1] / 100:.3f} "
        f"× 面积（平方米）+ {history['bias'][-1]:.2f}"
    )
    plot_training(areas_m2, prices, history)
    plt.show()


if __name__ == "__main__":
    main()
