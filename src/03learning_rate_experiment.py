"""在 f(x)=x² 上直观比较不同学习率的梯度下降轨迹。"""

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CONFIGURATIONS = [
    (0.05, "太小：收敛缓慢", "steelblue"),
    (0.4, "合适：稳步收敛", "seagreen"),
    (0.9, "偏大：震荡收敛", "orange"),
    (1.1, "过大：梯度爆炸", "crimson"),
]


def gradient_descent(start_x, learning_rate, steps=15, explosion_limit=500):
    """对 f(x)=x² 执行梯度下降，返回每一步的参数。"""
    trajectory = [start_x]
    x = start_x
    for _ in range(steps):
        gradient = 2 * x
        x -= learning_rate * gradient
        trajectory.append(x)
        if abs(x) > explosion_limit:
            break
    return np.asarray(trajectory)


def plot_parameter_trajectories(start_x=4.0):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (learning_rate, title, color) in zip(axes.flat, CONFIGURATIONS):
        trajectory = gradient_descent(start_x, learning_rate)
        axis.plot(trajectory, "o-", color=color, markersize=5)
        axis.axhline(0, color="gray", linestyle="--", alpha=0.5, label="最优点 x=0")
        axis.set_title(f"lr={learning_rate} — {title}")
        axis.set_xlabel("迭代次数")
        axis.set_ylabel("参数 x")
        axis.grid(True, alpha=0.3)
        axis.legend()
        print(
            f"lr={learning_rate:<4}: 执行 {len(trajectory) - 1:>2} 步，"
            f"最终 x={trajectory[-1]:.6f}, f(x)={trajectory[-1] ** 2:.6f}"
        )
    fig.suptitle("学习率对参数收敛的影响", fontsize=14, fontweight="bold")
    fig.tight_layout()


def plot_loss_curve_trajectories(start_x=4.0):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x_curve = np.linspace(-8, 8, 300)

    for axis, (learning_rate, title, color) in zip(axes.flat, CONFIGURATIONS):
        trajectory = gradient_descent(start_x, learning_rate, steps=10)
        visible = trajectory[np.abs(trajectory) <= 10]
        axis.plot(x_curve, x_curve**2, color="black", alpha=0.2, linewidth=2)
        axis.plot(visible, visible**2, "o", color=color, markersize=6)
        for current, following in zip(visible[:-1], visible[1:]):
            axis.annotate(
                "",
                xy=(following, following**2),
                xytext=(current, current**2),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
            )
        axis.set_title(f"lr={learning_rate} — {title}")
        axis.set_xlim(-9, 9)
        axis.set_ylim(-5, 70)
        axis.set_xlabel("x")
        axis.set_ylabel("f(x)=x²")
        axis.grid(True, alpha=0.3)
    fig.suptitle("学习率对损失曲面轨迹的影响", fontsize=14, fontweight="bold")
    fig.tight_layout()


def main():
    plot_parameter_trajectories()
    plot_loss_curve_trajectories()
    plt.show()


if __name__ == "__main__":
    main()
