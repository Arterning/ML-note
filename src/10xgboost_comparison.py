"""在加州房价数据集上比较随机森林与 XGBoost，并演示早停。"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_HOME = PROJECT_ROOT / "data" / "sklearn_data"


def load_data():
    # 首次运行会下载数据；缓存放在项目内，避免依赖用户主目录权限。
    try:
        dataset = fetch_california_housing(data_home=DATA_HOME)
        dataset_name = "加州房价"
    except OSError as error:
        # 数据源临时不可用时仍可完整演示两类模型的比较流程。
        print(f"加州房价数据下载失败（{error}）。")
        print("自动改用内置的糖尿病回归数据集。")
        dataset = load_diabetes()
        dataset_name = "糖尿病回归"

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
    )
    print(f"数据集: {dataset_name}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"特征: {dataset.feature_names}")
    return X_train, X_test, y_train, y_test


def timed_fit(model, X_train, y_train):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    return time.perf_counter() - start


def compare_models(X_train, X_test, y_train, y_test):
    random_forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    xgboost = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    rf_time = timed_fit(random_forest, X_train, y_train)
    xgb_time = timed_fit(xgboost, X_train, y_train)
    rf_prediction = random_forest.predict(X_test)
    xgb_prediction = xgboost.predict(X_test)

    print("\n" + "=" * 50)
    print(f"{'指标':<15} {'随机森林':>12} {'XGBoost':>12}")
    print("=" * 50)
    print(
        f"{'MSE':<15} {mean_squared_error(y_test, rf_prediction):>12.4f} "
        f"{mean_squared_error(y_test, xgb_prediction):>12.4f}"
    )
    print(
        f"{'R² Score':<15} {r2_score(y_test, rf_prediction):>12.4f} "
        f"{r2_score(y_test, xgb_prediction):>12.4f}"
    )
    print(f"{'训练时间(秒)':<15} {rf_time:>12.2f} {xgb_time:>12.2f}")


def demonstrate_early_stopping(X_train, X_test, y_train, y_test):
    model = XGBRegressor(
        n_estimators=10_000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=20,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    prediction = model.predict(X_test)
    print(f"\n最佳迭代轮数: {model.best_iteration}")
    print(f"早停模型 MSE: {mean_squared_error(y_test, prediction):.4f}")

    validation_rmse = model.evals_result()["validation_0"]["rmse"]
    plt.figure(figsize=(10, 5))
    plt.plot(validation_rmse, color="tomato", linewidth=2)
    plt.axvline(
        x=model.best_iteration,
        color="gray",
        linestyle="--",
        label=f"最佳轮数={model.best_iteration}",
    )
    plt.xlabel("迭代轮数（树的数量）")
    plt.ylabel("RMSE")
    plt.title("XGBoost 验证集误差")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    X_train, X_test, y_train, y_test = load_data()
    compare_models(X_train, X_test, y_train, y_test)
    demonstrate_early_stopping(X_train, X_test, y_train, y_test)
    plt.show()


if __name__ == "__main__":
    main()
