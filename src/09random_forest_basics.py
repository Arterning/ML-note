"""训练 Iris 随机森林，并比较 MDI 与排列特征重要性。"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def train_random_forest():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.3,
        random_state=42,
    )
    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return iris, model, X_test, y_test


def main():
    iris, model, X_test, y_test = train_random_forest()
    print(f"OOB 准确率: {model.oob_score_:.4f}")
    print(f"测试集准确率: {accuracy_score(y_test, model.predict(X_test)):.4f}")

    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=42,
        n_jobs=-1,
    )

    y_positions = np.arange(len(iris.feature_names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(y_positions, model.feature_importances_)
    axes[0].set_yticks(y_positions, iris.feature_names)
    axes[0].set_title("MDI 特征重要性")

    axes[1].barh(y_positions, permutation.importances_mean)
    axes[1].set_yticks(y_positions, iris.feature_names)
    axes[1].set_title("排列特征重要性")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
