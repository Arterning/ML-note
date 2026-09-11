"""使用训练时保留的测试集评估垃圾邮件模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .train import load_labeled_dataset


def evaluate(model_path: Path, dataset_path: Path, output_path: Path, threshold: float):
    model = joblib.load(model_path)
    dataframe = load_labeled_dataset(dataset_path)
    target = dataframe["target"].to_numpy()
    spam_index = list(model.classes_).index(1)
    spam_probability = model.predict_proba(dataframe)[:, spam_index]
    prediction = (spam_probability >= threshold).astype(int)

    matrix = confusion_matrix(target, prediction, labels=[0, 1])
    true_ham, false_spam, missed_spam, true_spam = matrix.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        target,
        prediction,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "sample_count": len(dataframe),
        "threshold": threshold,
        "accuracy": accuracy_score(target, prediction),
        "spam_precision": precision,
        "spam_recall": recall,
        "spam_f1": f1,
        "pr_auc": average_precision_score(target, spam_probability),
        "roc_auc": roc_auc_score(target, spam_probability),
        "ham_false_positive_rate": false_spam / max(1, true_ham + false_spam),
        "spam_false_negative_rate": missed_spam / max(1, missed_spam + true_spam),
        "confusion_matrix": matrix.tolist(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n分类报告：")
    print(
        classification_report(
            target,
            prediction,
            labels=[0, 1],
            target_names=["ham", "spam"],
            zero_division=0,
        )
    )

    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=["ham", "spam"]).plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
    )
    axis.set_title(f"Spam detection confusion matrix (threshold={threshold:.2f})")
    figure.tight_layout()
    image_path = output_path.with_name(f"{output_path.stem}_confusion_matrix.png")
    figure.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"评估结果: {output_path.resolve()}")
    print(f"混淆矩阵图片: {image_path.resolve()}")


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main():
    arguments = build_argument_parser().parse_args()
    if not 0 <= arguments.threshold <= 1:
        raise ValueError("--threshold 必须位于 [0, 1]。")
    evaluate(arguments.model, arguments.dataset, arguments.output, arguments.threshold)


if __name__ == "__main__":
    main()
