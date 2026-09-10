"""使用基尼系数决策树对 Iris 数据集分类并可视化。"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Songti SC"]
plt.rcParams["axes.unicode_minus"] = False

FEATURE_NAMES_CN = ["花萼长度(cm)", "花萼宽度(cm)", "花瓣长度(cm)", "花瓣宽度(cm)"]
TARGET_NAMES_CN = ["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"]


def show_dataset(iris):
    """以表格形式展示 Iris 数据集及类别数量。"""
    dataframe = pd.DataFrame(iris.data, columns=FEATURE_NAMES_CN)
    dataframe["最终分类结果"] = [TARGET_NAMES_CN[i] for i in iris.target]
    print("========== 数据集前 5 行 ==========")
    print(dataframe.head())
    print("\n========== 各分类的样本数量 ==========")
    print(dataframe["最终分类结果"].value_counts())


def main():
    iris = load_iris()
    show_dataset(iris)

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.3,
        random_state=42,
    )

    classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=42,
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(f"\n决策树模型的预测准确率: {accuracy_score(y_test, predictions):.2%}")

    plt.figure(figsize=(16, 12))
    tree.plot_tree(
        classifier,
        feature_names=FEATURE_NAMES_CN,
        class_names=TARGET_NAMES_CN,
        filled=True,
        rounded=True,
    )
    plt.title("Iris 决策树（Gini，max_depth=3）")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
