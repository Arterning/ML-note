"""比较线性回归、决策树和随机森林，演示如何缓解欠拟合。"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_HOME = PROJECT_ROOT / "data" / "sklearn_data"


def load_regression_data():
    """优先加载加州房价；远程数据不可用时回退到内置数据集。"""
    try:
        dataset = fetch_california_housing(data_home=DATA_HOME)
        name = "加州房价"
    except OSError as error:
        print(f"加州房价数据下载失败（{error}）。")
        print("自动改用内置的糖尿病回归数据集。")
        dataset = load_diabetes()
        name = "糖尿病回归"
    return name, dataset.data, dataset.target


def create_models():
    return {
        "线性回归": make_pipeline(StandardScaler(), LinearRegression()),
        "决策树（max_depth=5）": DecisionTreeRegressor(
            max_depth=5,
            random_state=42,
        ),
        "随机森林（100棵树）": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_prediction = model.predict(X_train)
        test_prediction = model.predict(X_test)
        train_r2 = r2_score(y_train, train_prediction)
        test_r2 = r2_score(y_test, test_prediction)
        results.append(
            {
                "模型": name,
                "训练 R²": train_r2,
                "测试 R²": test_r2,
                "训练 MSE": mean_squared_error(y_train, train_prediction),
                "测试 MSE": mean_squared_error(y_test, test_prediction),
                "R² 差距": train_r2 - test_r2,
            }
        )
    return pd.DataFrame(results)


def main():
    dataset_name, X, y = load_regression_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    results = evaluate_models(create_models(), X_train, X_test, y_train, y_test)
    print(f"\n数据集: {dataset_name}")
    print(f"训练集: {X_train.shape}，测试集: {X_test.shape}")
    print("\n各模型对比：")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
