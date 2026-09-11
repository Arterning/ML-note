"""训练 TF-IDF + LogisticRegression 垃圾邮件模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .model import NUMERIC_FEATURES, build_model


LABEL_MAPPING = {
    "ham": 0,
    "normal": 0,
    "正常": 0,
    "正常邮件": 0,
    "0": 0,
    "spam": 1,
    "junk": 1,
    "垃圾": 1,
    "垃圾邮件": 1,
    "1": 1,
}


def load_labeled_dataset(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path, keep_default_na=False)
    required = {"text", "label", "content_hash", "template_group", *NUMERIC_FEATURES}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"数据集缺少必要列: {sorted(missing)}")

    labels = dataframe["label"].astype(str).str.strip().str.lower().map(LABEL_MAPPING)
    dataframe = dataframe.loc[labels.notna()].copy()
    dataframe["target"] = labels.loc[labels.notna()].astype(int)
    dataframe.drop_duplicates(subset="content_hash", keep="first", inplace=True)
    dataframe["text"] = dataframe["text"].fillna("").astype(str)
    for column in NUMERIC_FEATURES:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(0)

    counts = dataframe["target"].value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError("训练数据必须同时包含 ham 和 spam 两个类别。")
    return dataframe.reset_index(drop=True)


def split_by_template(dataframe: pd.DataFrame, folds: int):
    groups = dataframe["template_group"].replace("", pd.NA).fillna(dataframe["content_hash"])
    group_counts = dataframe.assign(_group=groups).groupby("target")["_group"].nunique()
    max_folds = int(group_counts.min())
    if max_folds < 2:
        raise ValueError("每个类别至少需要两个不同模板，才能建立独立测试集。")
    actual_folds = min(folds, max_folds)
    splitter = StratifiedGroupKFold(
        n_splits=actual_folds,
        shuffle=True,
        random_state=42,
    )
    train_indices, test_indices = next(
        splitter.split(dataframe, dataframe["target"], groups=groups)
    )
    return dataframe.iloc[train_indices].copy(), dataframe.iloc[test_indices].copy(), actual_folds


def train(arguments):
    dataframe = load_labeled_dataset(arguments.dataset)
    train_data, test_data, actual_folds = split_by_template(dataframe, arguments.folds)
    model = build_model(
        max_word_features=arguments.max_word_features,
        max_char_features=arguments.max_char_features,
        regularization_c=arguments.c,
    )
    model.fit(train_data, train_data["target"])

    arguments.artifacts.mkdir(parents=True, exist_ok=True)
    model_path = arguments.artifacts / "model.joblib"
    test_path = arguments.artifacts / "test.csv"
    summary_path = arguments.artifacts / "split_summary.json"
    threshold_path = arguments.artifacts / "threshold.json"
    joblib.dump(model, model_path)
    test_data.drop(columns="target").to_csv(test_path, index=False, encoding="utf-8-sig")

    summary = {
        "all_labeled_unique": len(dataframe),
        "train_count": len(train_data),
        "test_count": len(test_data),
        "train_ham": int((train_data["target"] == 0).sum()),
        "train_spam": int((train_data["target"] == 1).sum()),
        "test_ham": int((test_data["target"] == 0).sum()),
        "test_spam": int((test_data["target"] == 1).sum()),
        "split_folds": actual_folds,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    threshold_path.write_text(
        json.dumps({"spam_threshold": 0.5}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"模型: {model_path.resolve()}")
    print(f"独立测试集: {test_path.resolve()}")
    print("下一步请运行 evaluate.py 评估测试集。")


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5, help="5 表示约 20% 用作测试")
    parser.add_argument("--max-word-features", type=int, default=80_000)
    parser.add_argument("--max-char-features", type=int, default=120_000)
    parser.add_argument("--c", type=float, default=1.0, help="逻辑回归正则化参数 C")
    return parser


def main():
    arguments = build_argument_parser().parse_args()
    if arguments.folds < 2:
        raise ValueError("--folds 必须至少为 2。")
    train(arguments)


if __name__ == "__main__":
    main()
