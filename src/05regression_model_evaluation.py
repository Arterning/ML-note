"""演示回归基线、常用指标、残差分析和交叉验证。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_HOME = PROJECT_ROOT / "data"


def load_regression_data():
    """加载项目中缓存的加州房价数据，必要时回退到内置数据。"""
    try:
        dataset = fetch_california_housing(data_home=DATA_HOME)
        dataset_name = "加州房价"
        target_unit = "10 万美元"
    except OSError as error:
        print(f"加州房价数据不可用（{error}）。改用糖尿病回归数据集。")
        dataset = load_diabetes()
        dataset_name = "糖尿病回归"
        target_unit = "目标值"

    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    print(f"数据集: {dataset_name}")
    print(f"样本数: {X.shape[0]}，特征数: {X.shape[1]}")
    print(f"目标范围: [{y.min():.2f}, {y.max():.2f}]，均值: {y.mean():.2f}")
    print(X.head())
    return X, y, target_unit


def train_baseline(X, y):
    """先划分数据，再在 Pipeline 内仅根据训练集进行标准化。"""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n训练集: {len(X_train)}，测试集: {len(X_test)}")
    print("前 10 个预测：")
    for actual, predicted in zip(y_test[:10], predictions[:10]):
        print(f"真实值: {actual:.4f}，预测值: {predicted:.4f}")
    return model, X_train, X_test, y_train, y_test, predictions


def evaluate_predictions(y_test, predictions, target_unit):
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n基线模型评估：")
    print(f"MAE:  {mae:.4f}（单位: {target_unit}）")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}（单位: {target_unit}）")
    print(f"R²:   {r2:.4f}（解释 {r2:.1%} 的方差）")


def plot_predictions_by_feature(X_test, y_test, predictions, target_unit):
    """使用同一批测试样本绘制特征、真实值与预测值，避免索引错位。"""
    sample_count = min(4000, len(X_test))
    X_sample = X_test.iloc[:sample_count]
    y_sample = y_test[:sample_count]
    prediction_sample = predictions[:sample_count]
    column_count = 4
    row_count = int(np.ceil(X_sample.shape[1] / column_count))
    fig, axes = plt.subplots(row_count, column_count, figsize=(20, 5 * row_count))
    axes = np.asarray(axes).ravel()

    for axis, feature_name in zip(axes, X_sample.columns):
        feature_values = X_sample[feature_name]
        axis.scatter(feature_values, y_sample, alpha=0.35, s=15, label="真实值")
        axis.scatter(
            feature_values,
            prediction_sample,
            alpha=0.35,
            s=15,
            marker="^",
            label="预测值",
        )
        axis.set_xlabel(feature_name)
        axis.set_ylabel(target_unit)
        axis.set_title(f"{feature_name} 与目标值")
        axis.grid(True, alpha=0.3)
        axis.legend()
    for axis in axes[X_sample.shape[1] :]:
        axis.axis("off")
    fig.tight_layout()
    print(f"已绘制前 {sample_count} 条测试样本的分特征对比图。")


def plot_residual_diagnostics(y_test, predictions):
    residuals = y_test - predictions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]

    axes[0].scatter(y_test, predictions, alpha=0.3, s=10, color="steelblue")
    axes[0].plot(limits, limits, "r--", linewidth=2, label="理想预测")
    axes[0].set_xlabel("真实值")
    axes[0].set_ylabel("预测值")
    axes[0].set_title("预测值与真实值")
    axes[0].legend()

    axes[1].scatter(predictions, residuals, alpha=0.3, s=10, color="steelblue")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("预测值")
    axes[1].set_ylabel("残差（真实值 - 预测值）")
    axes[1].set_title("残差与预测值")

    axes[2].hist(residuals, bins=50, edgecolor="black", color="steelblue", alpha=0.7)
    axes[2].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[2].set_xlabel("残差")
    axes[2].set_ylabel("数量")
    axes[2].set_title("残差分布")
    fig.tight_layout()

    print("\n残差统计：")
    print(f"均值: {residuals.mean():.4f}")
    print(f"标准差: {residuals.std():.4f}")
    print(f"偏度: {pd.Series(residuals).skew():.4f}")


def run_cross_validation(X, y):
    """将标准化放入 Pipeline，确保每一折仅根据其训练部分拟合。"""
    estimator = make_pipeline(StandardScaler(), LinearRegression())
    negative_mse = cross_val_score(
        estimator,
        X,
        y,
        cv=5,
        scoring="neg_mean_squared_error",
    )
    rmse_scores = np.sqrt(-negative_mse)
    print("\n每折 RMSE:")
    for index, score in enumerate(rmse_scores, start=1):
        print(f"Fold {index}: {score:.4f}")
    print(f"平均 RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    print(f"相对标准差: {rmse_scores.std() / rmse_scores.mean():.2%}")

    results = cross_validate(
        estimator,
        X,
        y,
        cv=5,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        return_train_score=True,
    )
    print(f"\n{'指标':<10} {'训练集':>12} {'验证集':>12} {'差距':>12}")
    print("-" * 50)
    for metric in ["rmse", "mae", "r2"]:
        train_mean = results[f"train_{metric}"].mean()
        validation_mean = results[f"test_{metric}"].mean()
        if metric != "r2":
            train_mean *= -1
            validation_mean *= -1
        print(
            f"{metric.upper():<10} {train_mean:>12.4f} "
            f"{validation_mean:>12.4f} {abs(train_mean - validation_mean):>12.4f}"
        )


def main():
    X, y, target_unit = load_regression_data()
    _, _, X_test, _, y_test, predictions = train_baseline(X, y)
    evaluate_predictions(y_test, predictions, target_unit)
    plot_predictions_by_feature(X_test, y_test, predictions, target_unit)
    plot_residual_diagnostics(y_test, predictions)
    run_cross_validation(X, y)
    plt.show()


if __name__ == "__main__":
    main()
