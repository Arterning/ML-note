# 机器学习笔记项目简介

运行 scripts 目录中的 .ipynb，推荐两种方式。

### 方式一：使用 VS Code

1. 用 VS Code 打开项目。
2. 打开 scripts 中的 .ipynb。
3. 点击右上角“选择内核”。
4. 选择项目环境：

.venv\Scripts\python.exe

当前环境已经安装了 ipykernel，因此可以直接运行单元格。

### 方式二：使用 Jupyter Lab

先把 Jupyter Lab 加为开发依赖：

uv add --dev jupyterlab

然后从项目根目录启动：

uv run jupyter lab scripts

浏览器会打开 Jupyter Lab，并以 scripts 目录作为入口。

如果不想把 Jupyter Lab写入项目依赖，也可以临时运行：

uv run --with jupyterlab jupyter lab scripts

以后新增 Python 依赖也建议使用：

uv add 包名

例如：

uv add lightgbm


- 原作其他博客  
https://juejin.cn/user/237150240001639/posts