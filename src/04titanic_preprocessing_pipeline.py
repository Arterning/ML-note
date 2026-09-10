"""用 ColumnTransformer 和 Pipeline 构建可复用的 Titanic 预处理流程。"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "titanic.csv"

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Pclass"]


def build_preprocessor():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def main():
    dataframe = pd.read_csv(DATA_PATH)
    dataframe.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe.drop(columns="Survived"),
        dataframe["Survived"],
        test_size=0.2,
        random_state=42,
        stratify=dataframe["Survived"],
    )
    preprocessor = build_preprocessor()
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    print("预处理流水线执行成功。")
    print(f"训练矩阵形状: {X_train_ready.shape}")
    print(f"测试矩阵形状: {X_test_ready.shape}")
    print("\n输出特征名：")
    print(preprocessor.get_feature_names_out())
    print("\n训练矩阵前 5 行：")
    print(X_train_ready[:5])
    print(f"\n标签形状: 训练 {y_train.shape}，测试 {y_test.shape}")


if __name__ == "__main__":
    main()
