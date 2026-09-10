# Iris 数据集与 `(150, 4)` 的含义

## 问题

`iris = load_iris()` 这个训练数据是表格数据吗？可以展示前几行给我看下吗？`(150, 4)` 是啥意思？

## 回答

是的，`load_iris()` 加载的是一个经典的表格数据集。每一行代表一朵鸢尾花，每一列代表一个测量特征。

可以在代码中加入：

```python
iris = load_iris()
X, y = iris.data, iris.target

print("特征名称：", iris.feature_names)
print("前 5 行特征：")
print(X[:5])

print("前 5 个类别编号：")
print(y[:5])
```

输出大致如下：

```text
特征名称：
[
  'sepal length (cm)',  # 花萼长度
  'sepal width (cm)',   # 花萼宽度
  'petal length (cm)',  # 花瓣长度
  'petal width (cm)'    # 花瓣宽度
]

前 5 行特征：
[[5.1 3.5 1.4 0.2]
 [4.9 3.0 1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.0 3.6 1.4 0.2]]

前 5 个类别编号：
[0 0 0 0 0]
```

`(150, 4)` 是 NumPy 数组 `X.shape` 的结果，其格式是：

```text
(行数, 列数)
(150, 4)
```

它表示：

- `150`：共有 150 个样本，也就是 150 朵鸢尾花。
- `4`：每朵花有 4 个测量特征。
- 因此，`X` 是一张 150 行、4 列的特征表。

标签数组 `y` 的形状是 `(150,)`，表示每朵花对应一个类别：

```text
0 = setosa（山鸢尾）
1 = versicolor（变色鸢尾）
2 = virginica（维吉尼亚鸢尾）
```

如果想以更直观的表格形式查看，可以转换为 Pandas DataFrame：

```python
import pandas as pd

df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y
df["species"] = [iris.target_names[i] for i in y]

print(df.head())
```

输出类似：

```text
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target species
0                5.1               3.5                1.4               0.2       0  setosa
1                4.9               3.0                1.4               0.2       0  setosa
2                4.7               3.2                1.3               0.2       0  setosa
3                4.6               3.1                1.5               0.2       0  setosa
4                5.0               3.6                1.4               0.2       0  setosa
```
