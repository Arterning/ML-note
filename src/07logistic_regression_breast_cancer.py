"""使用逻辑回归完成乳腺癌二分类，并展示常用评估指标。"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

TARGET_NAMES_CN = ["恶性 (0)", "良性 (1)"]


def load_and_show_data():
    """加载数据，并以 DataFrame 展示样本和统计信息。"""
    data = load_breast_cancer()
    X, y = data.data, data.target

    print(f"样本数: {X.shape[0]}，特征数: {X.shape[1]}")
    print(f"类别: {data.target_names}（0=恶性，1=良性）")
    print(f"各类别数量: 恶性={np.sum(y == 0)}，良性={np.sum(y == 1)}")

    dataframe = pd.DataFrame(X, columns=data.feature_names)
    dataframe["label"] = y
    print("\n前 5 行数据：")
    print(dataframe.head())
    print("\n各特征描述性统计：")
    print(dataframe.describe().round(3))
    return data, X, y


def prepare_data(X, y):
    """按类别分层划分数据，并仅使用训练集拟合标准化器。"""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train):
    """训练带默认 L2 正则化的逻辑回归模型。"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def plot_probability_distribution(probability, y_test):
    """按真实类别展示模型预测为良性的概率分布。"""
    malignant_probability = probability[y_test == 0]
    benign_probability = probability[y_test == 1]

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.hist(
        malignant_probability,
        bins=20,
        color="#e05c5c",
        alpha=0.7,
        label="真实：恶性 (0)",
        range=(0, 1),
    )
    axis.hist(
        benign_probability,
        bins=20,
        color="#4a90d9",
        alpha=0.7,
        label="真实：良性 (1)",
        range=(0, 1),
    )
    axis.axvline(0.5, color="#333333", linestyle="--", label="决策阈值 p=0.5")
    axis.set_xlabel(r"预测为良性的概率 $\hat{p}$")
    axis.set_ylabel("样本数")
    axis.set_title("逻辑回归预测概率分布（乳腺癌测试集）")
    axis.legend()
    fig.tight_layout()


def show_confusion_matrix(y_test, predictions):
    """绘制混淆矩阵，并明确医疗场景中的两类错误。"""
    matrix = confusion_matrix(y_test, predictions)
    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=TARGET_NAMES_CN,
    ).plot(ax=axis, colorbar=False, cmap="Blues")
    axis.set_title("混淆矩阵")
    fig.tight_layout()

    true_malignant, missed_malignant = matrix[0]
    false_alarm, true_benign = matrix[1]
    print("\n混淆矩阵：")
    print(matrix)
    print(f"恶性正确识别: {true_malignant}")
    print(f"恶性误判为良性（漏诊风险）: {missed_malignant}")
    print(f"良性误判为恶性（不必要复查）: {false_alarm}")
    print(f"良性正确识别: {true_benign}")


def plot_roc(probability, y_test):
    """绘制以良性类别（标签 1）为正类的 ROC 曲线。"""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_test,
        probability,
    )
    roc_auc = auc(false_positive_rate, true_positive_rate)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color="#4a90d9",
        linewidth=2,
        label=f"ROC（AUC={roc_auc:.4f}）",
    )
    axis.fill_between(false_positive_rate, true_positive_rate, alpha=0.08)
    axis.plot([0, 1], [0, 1], color="#999999", linestyle="--", label="随机猜测")

    for threshold, color, size in [(0.2, "gold", 50), (0.5, "red", 35), (0.8, "blue", 35)]:
        index = np.abs(thresholds - threshold).argmin()
        axis.scatter(
            false_positive_rate[index],
            true_positive_rate[index],
            color=color,
            s=size,
            label=(
                f"阈值≈{threshold:.1f} "
                f"(TPR={true_positive_rate[index]:.2f}, "
                f"FPR={false_positive_rate[index]:.2f})"
            ),
        )

    axis.set_xlabel("假正率 FPR（良性为正类）")
    axis.set_ylabel("真正率 TPR（良性召回率）")
    axis.set_title("ROC 曲线")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=9)
    fig.tight_layout()
    print(f"\nAUC: {roc_auc:.4f}")


def main():
    _, X, y = load_and_show_data()
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    benign_probability = probabilities[:, 1]

    print("\n训练完成")
    print(f"训练集准确率: {model.score(X_train, y_train):.4f}")
    print(f"测试集准确率: {accuracy_score(y_test, predictions):.4f}")
    print("\n前 5 个样本的 [恶性概率, 良性概率]：")
    print(np.round(probabilities[:5], 4))
    print("对应预测标签:", predictions[:5])

    print("\n分类报告：")
    print(classification_report(y_test, predictions, target_names=TARGET_NAMES_CN))

    plot_probability_distribution(benign_probability, y_test)
    show_confusion_matrix(y_test, predictions)
    plot_roc(benign_probability, y_test)
    plt.show()


if __name__ == "__main__":
    main()
