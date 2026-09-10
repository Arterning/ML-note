"""分析随机森林中的单棵树，以及手动推导 MDI 特征重要性。"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

FEATURE_NAMES_CN = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]
TARGET_NAMES_CN = ["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"]
TREE_INDICES = [0, 25, 50, 75, 99]


def train_forest():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.3,
        random_state=42,
    )
    forest = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    ).fit(X_train, y_train)
    return forest, X_train, X_test, y_train, y_test


def plot_selected_trees(forest, X_train, X_test, y_train, y_test):
    selected = [forest.estimators_[i] for i in TREE_INDICES]
    print(f"选取的树索引: {TREE_INDICES}")
    print(f"最大深度: {[model.tree_.max_depth for model in selected]}")
    print(f"叶子数: {[model.tree_.n_leaves for model in selected]}")

    fig, axes = plt.subplots(2, 3, figsize=(28, 18))
    for axis, tree_index, estimator in zip(axes.flat, TREE_INDICES, selected):
        train_accuracy = accuracy_score(y_train, estimator.predict(X_train))
        test_accuracy = accuracy_score(y_test, estimator.predict(X_test))
        tree.plot_tree(
            estimator,
            feature_names=FEATURE_NAMES_CN,
            class_names=TARGET_NAMES_CN,
            filled=True,
            rounded=True,
            ax=axis,
            fontsize=9,
        )
        axis.set_title(
            f"第 {tree_index + 1} 棵树 | 训练 {train_accuracy:.2%} | 测试 {test_accuracy:.2%}"
        )
    axes.flat[-1].axis("off")
    fig.suptitle("随机森林中均匀选取的 5 棵决策树", fontsize=18)
    fig.tight_layout()
    return selected


def compare_tree_importances(forest, selected):
    per_tree = np.array([estimator.feature_importances_ for estimator in selected])
    forest_importance = forest.feature_importances_
    table = pd.DataFrame(
        per_tree.T,
        index=FEATURE_NAMES_CN,
        columns=[f"第 {i + 1} 棵树" for i in TREE_INDICES],
    )
    table["5树平均"] = per_tree.mean(axis=0)
    table["森林整体"] = forest_importance
    print("\n各树的 MDI 特征重要性：")
    print(table.round(4))

    ranks = pd.DataFrame(
        [rankdata(-importance, method="min") for importance in per_tree],
        columns=FEATURE_NAMES_CN,
        index=[f"第 {i + 1} 棵树" for i in TREE_INDICES],
    )
    ranks.loc["森林整体"] = rankdata(-forest_importance, method="min")
    print("\n特征排名（1 表示最重要）：")
    print(ranks.astype(int))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 4))
    for axis, tree_index, importance in zip(axes.flat, TREE_INDICES, per_tree):
        order = np.argsort(importance)
        axis.barh(
            [FEATURE_NAMES_CN[i] for i in order],
            importance[order],
            color=[colors[i] for i in order],
        )
        axis.set_title(f"第 {tree_index + 1} 棵树 MDI")

    axis = axes.flat[-1]
    positions = np.arange(4)
    order = np.argsort(per_tree.mean(axis=0))
    axis.bar(positions - 0.18, per_tree.mean(axis=0)[order], 0.36, label="5树平均")
    axis.bar(positions + 0.18, forest_importance[order], 0.36, label="森林整体")
    axis.set_xticks(positions, [FEATURE_NAMES_CN[i] for i in order])
    axis.set_title("5 树平均与森林整体 MDI")
    axis.legend()
    fig.tight_layout()


def derive_mdi(estimator):
    """使用 sklearn 相同的加权不纯度下降公式重算一棵树的 MDI。"""
    structure = estimator.tree_
    contributions = np.zeros(len(FEATURE_NAMES_CN))
    print("\n第 1 棵树各分裂节点的不纯度贡献：")
    for node in range(structure.node_count):
        left = structure.children_left[node]
        right = structure.children_right[node]
        if left == -1:
            continue
        feature = structure.feature[node]
        contribution = (
            structure.weighted_n_node_samples[node] * structure.impurity[node]
            - structure.weighted_n_node_samples[left] * structure.impurity[left]
            - structure.weighted_n_node_samples[right] * structure.impurity[right]
        )
        contributions[feature] += contribution
        print(f"节点 {node:>2} | {FEATURE_NAMES_CN[feature]} | 贡献={contribution:.6f}")

    manual = contributions / contributions.sum()
    print("\n手动推导 MDI: ", np.round(manual, 6))
    print("sklearn MDI:  ", np.round(estimator.feature_importances_, 6))
    print("是否一致:     ", np.allclose(manual, estimator.feature_importances_))

    print("\nBootstrap 样本计数：")
    print(f"根节点唯一训练样本数: {structure.n_node_samples[0]}")
    print(f"根节点加权样本数: {structure.weighted_n_node_samples[0]:.0f}")
    print("MDI 计算应使用 weighted_n_node_samples。")

    # sklearn 1.7 中，分类树的 value 轴顺序为 [节点, 输出, 类别]，值为类别比例。
    root_proportions = structure.value[0, 0]
    print("\ntree_.value 根节点类别比例:", np.round(root_proportions, 4))
    print("比例之和:", root_proportions.sum())
    print(
        "按 Bootstrap 权重换算的类别计数:",
        np.round(root_proportions * structure.weighted_n_node_samples[0], 2),
    )


def main():
    forest, X_train, X_test, y_train, y_test = train_forest()
    selected = plot_selected_trees(forest, X_train, X_test, y_train, y_test)
    compare_tree_importances(forest, selected)
    derive_mdi(selected[0])
    plt.show()


if __name__ == "__main__":
    main()
