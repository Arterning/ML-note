# 垃圾邮件检测：TF-IDF + LogisticRegression

这个目录提供一套可以直接运行的垃圾邮件训练与测试流程：

```text
EML 文件
  → 安全解析主题、正文和邮件头
  → 生成 CSV 数据清单
  → 词级/字符级 TF-IDF + 结构化特征
  → LogisticRegression
  → 独立测试集评估或预测新邮件
```

## 目录结构

```text
spam_email_detection/
├── spam_email_detector/
│   ├── __init__.py
│   ├── email_parser.py       # EML 解析和结构化特征提取
│   ├── model.py              # TF-IDF + LogisticRegression 流水线
│   ├── prepare_eml.py        # EML → CSV
│   ├── train.py              # 划分数据、训练和保存模型
│   ├── evaluate.py           # 在保留测试集上评估
│   └── predict.py            # 预测新 EML
├── artifacts/                # 训练后生成模型与评估文件
└── data/                     # 建议存放生成的数据清单
```

原始邮件可以放在项目外，通过命令行参数传入。程序不会打开或执行附件，只会统计附件信息。

## 1. 准备 EML

### 方式 A：已经按文件夹分好标签

目录可以组织为：

```text
your_emails/
├── ham/
│   ├── normal_001.eml
│   └── normal_002.eml
└── spam/
    ├── spam_001.eml
    └── spam_002.eml
```

程序会识别路径中的以下目录名：

- 正常邮件：`ham`、`normal`、`正常邮件`
- 垃圾邮件：`spam`、`junk`、`垃圾邮件`

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m spam_email_detection.spam_email_detector.prepare_eml `
  --input "D:\your_emails" `
  --output spam_email_detection\data\emails.csv
```

### 方式 B：邮件完全没有标签

直接运行同一条命令。无法从目录推断标签的邮件，其 `label` 会留空：

```powershell
.\.venv\Scripts\python.exe -m spam_email_detection.spam_email_detector.prepare_eml `
  --input "D:\unlabeled_emails" `
  --output spam_email_detection\data\emails_to_label.csv
```

然后打开 CSV，只需填写 `label` 列：

```text
ham   = 正常邮件
spam  = 垃圾邮件
```

不确定的邮件可以保持空白；训练时会自动忽略未标注行。不要修改 `email_id`、`text`、`content_hash` 和其他特征列。

CSV 中的正文可能包含敏感信息，应限制文件访问权限，不要上传到公共代码仓库。

## 2. 训练模型

```powershell
.\.venv\Scripts\python.exe -m spam_email_detection.spam_email_detector.train `
  --dataset spam_email_detection\data\emails.csv `
  --artifacts spam_email_detection\artifacts
```

训练过程会：

1. 忽略未标注或无法识别标签的记录。
2. 按 `content_hash` 删除完全重复内容。
3. 按 `template_group` 分组划分训练集和测试集，降低相似模板泄漏。
4. 仅使用训练集拟合 TF-IDF 和模型。
5. 保存模型、独立测试集、数据划分摘要和默认阈值。

生成文件：

```text
artifacts/model.joblib
artifacts/test.csv
artifacts/split_summary.json
artifacts/threshold.json
```

可以调整测试折数、最大词汇量和正则化强度：

```powershell
.\.venv\Scripts\python.exe -m spam_email_detection.spam_email_detector.train `
  --dataset spam_email_detection\data\emails.csv `
  --artifacts spam_email_detection\artifacts `
  --folds 5 `
  --max-word-features 80000 `
  --max-char-features 120000 `
  --c 2.0
```

## 3. 测试和评估

```powershell
.\.venv\Scripts\python.exe -m spam_email_detection.spam_email_detector.evaluate `
  --model spam_email_detection\artifacts\model.joblib `
  --dataset spam_email_detection\artifacts\test.csv `
  --output spam_email_detection\artifacts\evaluation.json
```

输出包括：

- Accuracy
- Spam Precision
- Spam Recall
- Spam F1
- PR-AUC
- ROC-AUC
- 混淆矩阵
- 正常邮件误判率
- 垃圾邮件漏检率

同时生成 `evaluation_confusion_matrix.png`。

正常邮件误判为垃圾邮件的成本通常较高，所以不能只看 Accuracy，应重点观察 `ham_false_positive_rate` 和 spam precision。

## 4. 预测新的 EML

预测单封邮件：

```powershell
.\.venv\Scripts\python.exe -m spam_email_detection.spam_email_detector.predict `
  --model spam_email_detection\artifacts\model.joblib `
  --input "D:\new_mail\message.eml"
```

批量预测一个目录：

```powershell
.\.venv\Scripts\python.exe -m spam_email_detection.spam_email_detector.predict `
  --model spam_email_detection\artifacts\model.joblib `
  --input "D:\new_mail" `
  --output spam_email_detection\artifacts\predictions.csv
```

使用自定义阈值：

```powershell
.\.venv\Scripts\python.exe -m spam_email_detection.spam_email_detector.predict `
  --model spam_email_detection\artifacts\model.joblib `
  --input "D:\new_mail" `
  --threshold 0.85
```

阈值越高，正常邮件被误判为垃圾邮件的概率通常越低，但可能漏掉更多垃圾邮件。

## 5. 模型使用的特征

文本特征：

- 主题 + 正文的词级 TF-IDF，`ngram_range=(1, 2)`。
- 主题 + 正文的字符级 TF-IDF，`ngram_range=(2, 5)`。

结构化特征：

- 正文长度。
- 主题长度。
- URL 数量。
- 收件人数量。
- 附件数量。
- 可疑附件数量。
- 感叹号数量。
- 大写字符比例。
- 是否包含 HTML。
- 发件人与 Reply-To 域名是否不同。

最终分类器为带 L2 正则化的 `LogisticRegression`，并启用 `class_weight="balanced"` 处理类别不平衡。

## 6. 速度建议

- 启动服务时加载一次模型，让模型常驻内存。
- 批量预测目录比每封邮件启动一次 Python 更快。
- 默认不提取附件内容，避免解码大文件。
- 可以通过 `--max-body-chars` 限制正文长度。
- DNS、黑名单和远程信誉查询应放在异步流程中，不要阻塞主模型。
- 实际上线前，用自己的 EML 运行吞吐和延迟基准测试。

## 7. 重要限制

- 未标注邮件不能直接用于监督训练，至少需要一批可靠的 `ham/spam` 标签。
- 聚类或规则生成的伪标签应与人工标签区分。
- 最终测试集应由人工确认，不应全部来自自动伪标签。
- 相似群发模板不能同时出现在训练集和测试集中，否则指标会虚高。
- 模型只能反映训练数据中的规律，需要持续收集误判样本并定期重新训练。
