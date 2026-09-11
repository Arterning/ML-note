# 垃圾邮件模型扩展为多类别的设计建议

## 问题

如果将邮件类型扩展为以下类别，现有的 TF-IDF + LogisticRegression 项目是否需要修改？

```text
正常邮件
营销邮件
普通垃圾邮件
钓鱼邮件
恶意附件邮件
```

## 结论

需要修改。当前代码是严格的二分类设计：

```text
ham  = 0
spam = 1
```

如果扩展为五种邮件类型，训练、评估和预测代码都需要从二分类改成多分类。不过在修改代码前，建议先确定标签体系，因为这五类邮件不一定互斥。

## 一、类别可能发生重叠

一封邮件可能同时属于：

```text
钓鱼邮件 + 带恶意附件
```

营销邮件也可能同时符合“普通垃圾邮件”的定义。

因此，可以采用以下两种设计方案。

## 二、方案 A：每封邮件只有一个类别

可以规定类别优先级：

```text
恶意附件 > 钓鱼 > 普通垃圾 > 营销 > 正常
```

例如，一封钓鱼邮件带有恶意附件时，最终标记为：

```text
malicious_attachment
```

标签可以定义为：

```text
normal                正常邮件
marketing             营销邮件
spam                  普通垃圾邮件
phishing              钓鱼邮件
malicious_attachment  恶意附件邮件
```

优点：

- 代码和训练流程相对简单。
- 逻辑回归可以直接完成多分类。
- 每封邮件只有一个分类结果，容易展示和统计。

缺点：

- 会丢失类别重叠信息。
- 无法同时表示“钓鱼 + 恶意附件”。

## 三、方案 B：主分类 + 安全标签（推荐）

将邮件性质和安全威胁分开：

```text
主分类：
normal
marketing
spam

安全标签：
is_phishing
has_malicious_attachment
```

一封邮件的标签可以表示为：

```json
{
  "category": "spam",
  "is_phishing": true,
  "has_malicious_attachment": true
}
```

模型结构：

```text
邮件
 ├── 主分类模型：正常 / 营销 / 普通垃圾
 ├── 钓鱼检测模型：是 / 否
 └── 恶意附件检测：是 / 否
```

这种设计更符合实际邮件安全系统，因为邮件内容类别和安全威胁可以同时存在。

## 四、现有代码需要修改的位置

### 1. 标签推断

当前 `email_parser.py` 只识别：

```text
ham
spam
```

如果使用单一多分类标签，需要增加文件夹映射，例如：

```text
emails/
├── normal/
├── marketing/
├── spam/
├── phishing/
└── malicious_attachment/
```

### 2. 训练标签

当前 `train.py` 的标签映射只有：

```python
ham -> 0
spam -> 1
```

多分类时可以扩展为：

```python
LABEL_MAPPING = {
    "normal": 0,
    "marketing": 1,
    "spam": 2,
    "phishing": 3,
    "malicious_attachment": 4,
}
```

也可以直接使用字符串标签，让 scikit-learn 自动处理。

`LogisticRegression` 本身支持多分类，因此核心模型不需要更换。

### 3. 预测结果

当前预测代码只输出：

```text
spam_probability
prediction = ham / spam
```

多分类后，需要输出每种类别的概率：

```text
normal_probability
marketing_probability
spam_probability
phishing_probability
malicious_attachment_probability
prediction
```

例如：

```text
normal:                0.02
marketing:             0.08
spam:                  0.10
phishing:              0.75
malicious_attachment:  0.05

最终类别: phishing
```

### 4. 评估代码

当前评估指标以垃圾邮件为正类，包括：

```text
spam precision
spam recall
spam F1
正常邮件误判率
垃圾邮件漏检率
```

多分类后，应改为：

- 每个类别的 Precision、Recall 和 F1。
- Macro F1。
- Weighted F1。
- 多分类混淆矩阵。
- 每一类的 One-vs-Rest PR-AUC 或 ROC-AUC。
- 钓鱼邮件漏检率。
- 恶意附件邮件漏检率。

安全类邮件应重点关注：

```text
phishing recall
malicious_attachment recall
```

### 5. 数据划分

现有的模板分组划分可以继续使用，但必须保证：

- 每个类别都有足够多的不同邮件模板。
- 同一个模板不能跨训练集和测试集。
- 每个类别都出现在训练集和测试集中。
- 少数类别不能只有几封邮件。

## 五、恶意附件不能只靠邮件正文判断

当前解析器只提取：

- 附件数量。
- 附件扩展名。
- 可疑扩展名数量。

这些特征可以发现：

```text
.exe
.js
.vbs
.ps1
.scr
.bat
```

但不能真正判断附件是否恶意。例如：

```text
invoice.pdf
document.docx
archive.zip
```

扩展名看起来正常，内容仍然可能是恶意的。

更加稳妥的附件检测流程是：

```text
附件
  ├── 文件扩展名
  ├── MIME 类型
  ├── 文件大小
  ├── SHA-256 哈希
  ├── 是否为加密压缩包
  ├── Office 是否包含宏
  ├── PDF 是否包含脚本
  └── 杀毒引擎或沙箱检测结果
```

不要让 Python 程序打开或执行附件。可以提取附件字节计算哈希，再交给杀毒引擎或隔离沙箱检查。

因此，“恶意附件邮件”最好由专门的附件检测模块判断，而不是只依赖 TF-IDF。

## 六、推荐的最终架构

```text
EML 解析
  ↓
规则快速检查
  ├── 发件人、域名、URL
  └── 附件类型和安全扫描
  ↓
文本模型
  ├── 正常
  ├── 营销
  └── 普通垃圾
  ↓
独立安全模型
  ├── 钓鱼概率
  └── 恶意附件检测结果
  ↓
综合决策
```

最终结果可以表示为：

```json
{
  "category": "marketing",
  "category_probability": 0.91,
  "is_phishing": false,
  "phishing_probability": 0.04,
  "has_malicious_attachment": false,
  "attachment_scan": "clean"
}
```

## 七、最终建议

如果只是学习和演示，可以将现有代码改为五分类逻辑回归。

如果准备实际使用，更推荐：

```text
三分类主模型
    正常 / 营销 / 普通垃圾

+ 钓鱼二分类模型

+ 独立附件安全扫描
```

这种分层设计既能保留类别重叠信息，也能对钓鱼和恶意附件采用更严格、更专业的检测方式。

目前暂不修改现有二分类项目代码，等标签体系确定后再进行扩展。
