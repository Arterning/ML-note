"""使用训练好的模型预测单个 EML 或整个目录。"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .email_parser import parse_eml
from .prepare_eml import iter_eml_files


def predict_emails(
    model_path: Path,
    input_path: Path,
    threshold: float,
    output_path: Path | None,
    max_body_chars: int,
):
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    model = joblib.load(model_path)
    input_root = input_path if input_path.is_dir() else input_path.parent
    records = [
        parse_eml(path, input_root=input_root, max_body_chars=max_body_chars)
        for path in iter_eml_files(input_path)
    ]
    if not records:
        raise ValueError("输入路径中没有找到 EML 文件。")

    dataframe = pd.DataFrame(records)
    spam_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(dataframe)[:, spam_index]
    result = dataframe[["email_id", "source_path", "subject", "sender_domain"]].copy()
    result["spam_probability"] = probabilities
    result["prediction"] = ["spam" if value >= threshold else "ham" for value in probabilities]
    result["threshold"] = threshold
    result.sort_values("spam_probability", ascending=False, inplace=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"预测结果: {output_path.resolve()}")
    else:
        with pd.option_context("display.max_colwidth", 60, "display.width", 160):
            print(result.to_string(index=False))
    print("\n预测数量:", len(result))
    print(result["prediction"].value_counts().to_string())


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-body-chars", type=int, default=50_000)
    return parser


def main():
    arguments = build_argument_parser().parse_args()
    if not 0 <= arguments.threshold <= 1:
        raise ValueError("--threshold 必须位于 [0, 1]。")
    predict_emails(
        arguments.model,
        arguments.input,
        arguments.threshold,
        arguments.output,
        arguments.max_body_chars,
    )


if __name__ == "__main__":
    main()
