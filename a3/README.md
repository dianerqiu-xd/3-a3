# A3 语义分析综合测试平台

这是 Week 4 随堂 Vibe 实验的 Streamlit Web 应用，集成四类语义表示模型：

- 模块 1：TF-IDF 关键词提取与 LSA 二维降维可视化。
- 模块 2：Word2Vec 实时训练，并对比 CBOW 与 Skip-Gram。
- 模块 3：使用 `gensim.downloader` 加载 `glove-twitter-25`，完成词类比和词义相似度测试。
- 模块 4：FastText OOV 鲁棒性测试，以及平均池化版本的简单 Sent2Vec。

## 本地运行

```bash
cd /Users/qiudianer/周三作业/a3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

打开浏览器访问：

```text
http://localhost:8501
```

第一次进入 GloVe 标签页时会下载 `glove-twitter-25`。如果网络不可用，程序会自动切换到内置的小型演示向量，保证页面不崩溃；部署到 Streamlit Cloud 后通常可以正常下载预训练模型。

## 上传 GitHub

```bash
cd /Users/qiudianer/周三作业
git add a3
git commit -m "Add A3 semantic analysis Streamlit app"
git push
```

## Streamlit Community Cloud 部署

1. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)。
2. 选择 `New app`。
3. 选择你的 GitHub 仓库和分支。
4. `Main file path` 填写：

```text
a3/streamlit_app.py
```

5. 点击 `Deploy`，构建完成后会得到形如 `https://你的应用名.streamlit.app` 的公开 URL。

## 提交文件

- `streamlit_app.py`：四个模块的 Streamlit 页面。
- `semantic_core.py`：语义表示、训练、相似度和类比计算核心代码。
- `requirements.txt`：部署依赖。
- `A3_语义表示实验报告.pdf`：可直接提交的实验报告 PDF。
- `A3_实验报告_可打印.html`：可用浏览器打印为 PDF 的展示页面 HTML。
- `make_report.py`：重新生成报告 PDF 和报告配图的脚本。
- `assets/`：报告中使用的界面示意图。
