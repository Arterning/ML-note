"""递归解析 EML 文件并生成可标注、可训练的 CSV。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .email_parser import parse_eml


FIELD_NAMES = [
    "email_id", "source_path", "label", "subject", "body", "text",
    "sender_domain", "content_hash", "template_group", "text_length",
    "subject_length", "url_count", "recipient_count", "attachment_count",
    "suspicious_attachment_count", "exclamation_count", "uppercase_ratio",
    "has_html", "reply_to_mismatch",
]


def iter_eml_files(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() != ".eml":
            raise ValueError("输入文件必须使用 .eml 扩展名。")
        yield input_path
        return
    yield from sorted(path for path in input_path.rglob("*.eml") if path.is_file())


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    max_body_chars: int = 50_000,
    max_file_mb: float = 25,
):
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_root = input_path if input_path.is_dir() else input_path.parent
    max_file_bytes = int(max_file_mb * 1024 * 1024)
    processed = failed = skipped_large = labeled = 0

    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for eml_path in iter_eml_files(input_path):
            if eml_path.stat().st_size > max_file_bytes:
                skipped_large += 1
                print(f"跳过超大文件: {eml_path}", file=sys.stderr)
                continue
            try:
                record = parse_eml(
                    eml_path,
                    input_root=input_root,
                    max_body_chars=max_body_chars,
                )
                writer.writerow(record)
                processed += 1
                labeled += bool(record["label"])
            except Exception as error:
                failed += 1
                print(f"解析失败 {eml_path}: {error}", file=sys.stderr)

    print(f"输出文件: {output_path.resolve()}")
    print(f"成功解析: {processed}")
    print(f"已自动推断标签: {labeled}")
    print(f"标签留空: {processed - labeled}")
    print(f"超大文件跳过: {skipped_large}")
    print(f"解析失败: {failed}")


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="EML 文件或目录")
    parser.add_argument("--output", required=True, type=Path, help="输出 CSV 路径")
    parser.add_argument("--max-body-chars", type=int, default=50_000)
    parser.add_argument("--max-file-mb", type=float, default=25)
    return parser


def main():
    arguments = build_argument_parser().parse_args()
    prepare_dataset(
        arguments.input,
        arguments.output,
        max_body_chars=arguments.max_body_chars,
        max_file_mb=arguments.max_file_mb,
    )


if __name__ == "__main__":
    main()
