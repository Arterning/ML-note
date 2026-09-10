"""对 Titanic 数据进行划分、探索性分析、缺失值和异常值处理。"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "titanic.csv"


def load_and_split_data():
    dataframe = pd.read_csv(DATA_PATH)
    dataframe.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    X = dataframe.drop(columns="Survived")
    y = dataframe["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    # 后续会原地修改，因此显式复制，避免链式赋值问题。
    return dataframe, X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()


def show_overview(dataframe, X_train, X_test, y_train, y_test):
    print(f"数据集形状: {dataframe.shape}")
    print("\n前 5 行数据：")
    print(dataframe.head())
    print("\n训练集数据类型：")
    X_train.info()

    missing = X_train.isna().sum()
    missing_table = pd.DataFrame(
        {
            "缺失数": missing,
            "缺失率%": (missing / len(X_train) * 100).round(1),
        }
    )
    print("\n包含缺失值的列：")
    print(missing_table.loc[missing_table["缺失数"] > 0])
    print("\n数值特征统计：")
    print(X_train.describe().round(2))
    print(f"\n训练集: {len(X_train)}，测试集: {len(X_test)}")
    print(f"训练集存活率: {y_train.mean():.2%}")
    print(f"测试集存活率: {y_test.mean():.2%}")


def plot_eda(X_train, y_train):
    train_data = X_train.copy()
    train_data["Survived"] = y_train
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    y_train.value_counts().sort_index().plot.bar(
        ax=axes[0, 0], color=["#e74c3c", "#2ecc71"]
    )
    axes[0, 0].set_title("存活率分布")
    axes[0, 0].set_xticklabels(["Dead (0)", "Survived (1)"], rotation=0)
    X_train["Age"].hist(bins=30, ax=axes[0, 1], color="steelblue", edgecolor="black")
    axes[0, 1].set_title("年龄分布")
    X_train["Fare"].hist(bins=30, ax=axes[0, 2], color="coral", edgecolor="black")
    axes[0, 2].set_title("票价分布")
    sns.barplot(data=train_data, x="Sex", y="Survived", ax=axes[1, 0])
    axes[1, 0].set_title("性别与存活率")
    sns.barplot(data=train_data, x="Pclass", y="Survived", ax=axes[1, 1])
    axes[1, 1].set_title("船票等级与存活率")
    numeric_columns = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    sns.heatmap(
        train_data[numeric_columns].corr(),
        annot=True,
        cmap="RdBu_r",
        center=0,
        ax=axes[1, 2],
    )
    axes[1, 2].set_title("相关性热力图")
    fig.tight_layout()


def remove_duplicate_rows(X_train, y_train):
    duplicate_mask = X_train.duplicated()
    print(f"\n重复行数: {duplicate_mask.sum()}")
    return X_train.loc[~duplicate_mask].copy(), y_train.loc[~duplicate_mask].copy()


def fill_missing_values(X_train, X_test):
    """所有填充器只在训练集上拟合，再应用至测试集。"""
    age_imputer = SimpleImputer(strategy="median")
    X_train[["Age"]] = age_imputer.fit_transform(X_train[["Age"]])
    X_test[["Age"]] = age_imputer.transform(X_test[["Age"]])

    embarked_imputer = SimpleImputer(strategy="most_frequent")
    X_train[["Embarked"]] = embarked_imputer.fit_transform(X_train[["Embarked"]])
    X_test[["Embarked"]] = embarked_imputer.transform(X_test[["Embarked"]])

    for frame in (X_train, X_test):
        frame["HasCabin"] = frame["Cabin"].notna().astype(int)
        frame.drop(columns="Cabin", inplace=True)

    print(f"Age 训练集中位数: {age_imputer.statistics_[0]:.1f}")
    print(f"Embarked 训练集众数: {embarked_imputer.statistics_[0]}")
    return X_train, X_test


def clip_fare_outliers(X_train, X_test, factor=1.5):
    first_quartile = X_train["Fare"].quantile(0.25)
    third_quartile = X_train["Fare"].quantile(0.75)
    iqr = third_quartile - first_quartile
    lower = max(0, first_quartile - factor * iqr)
    upper = third_quartile + factor * iqr
    outlier_count = ((X_train["Fare"] < lower) | (X_train["Fare"] > upper)).sum()
    print(f"Fare IQR 合理范围: [{lower:.2f}, {upper:.2f}]")
    print(f"截断前异常值数量: {outlier_count}")
    X_train["Fare"] = X_train["Fare"].clip(lower, upper)
    X_test["Fare"] = X_test["Fare"].clip(lower, upper)
    return X_train, X_test


def plot_boxplots(X_train):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, column in zip(axes, ["Age", "Fare", "SibSp"]):
        X_train.boxplot(column=column, ax=axis)
        axis.set_title(f"{column} Boxplot")
    fig.tight_layout()


def main():
    dataframe, X_train, X_test, y_train, y_test = load_and_split_data()
    show_overview(dataframe, X_train, X_test, y_train, y_test)
    plot_eda(X_train, y_train)
    X_train, y_train = remove_duplicate_rows(X_train, y_train)
    X_train, X_test = fill_missing_values(X_train, X_test)
    X_train, X_test = clip_fare_outliers(X_train, X_test)
    plot_boxplots(X_train)
    print("\n清洗后训练集各列缺失值：")
    print(X_train.isna().sum())
    print(f"清洗后形状: 训练集 {X_train.shape}，测试集 {X_test.shape}")
    plt.show()


if __name__ == "__main__":
    main()
