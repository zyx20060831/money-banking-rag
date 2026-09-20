# 货币银行学报告问答

面向货币银行学课程的银行年报与货币政策报告问答工具。使用 pdfplumber 提取 PDF 文字（无文字层的扫描件用 RapidOCR 自动补全）、本地 BM25 检索证据，通过可配置的大模型生成带引用的回答，并提供 Streamlit 网页与命令行。

## 项目材料

- 4 份报告：宁波银行与中国银行 2025 年年报、2024 年及 2025 年第四季度中国货币政策执行报告。
- 70 道预设问题：40 道银行经营题、30 道宏观金融题。
- 支持手动提问与题库批量运行；目前没有自动出题功能。

## 快速启动

需要 Python 3.10 或更高版本。在项目根目录运行：

```powershell
python -m pip install -r requirements.txt
python -m annual_report_rag build --pdf-dir . --data-dir data
python -m streamlit run app.py
```

索引在本地生成，`data/`、运行输出 `output/` 与临时文件 `tmp/` 不纳入版本控制。

未配置模型时，项目仅展示检索原文和物理页码。需要生成回答时，复制 `.env.example` 为 `.env` 并填入模型服务信息，程序启动时会自动读取项目根目录的 `.env`：

```
RAG_BASE_URL=https://你的服务地址/v1
RAG_MODEL=你的模型名
RAG_API_KEY=你的密钥
```

也可以不改 `.env`，直接在启动程序前设置环境变量（环境变量优先级更高）。请使用兼容 Chat Completions 的模型服务，本地模型可按服务要求省略密钥。`.env` 已被 `.gitignore` 忽略，不要将真实密钥提交到仓库。

## 命令行示例

```powershell
python -m annual_report_rag documents
python -m annual_report_rag search --document "中国银行_2025" --query "银行的净息差是多少？"
python -m annual_report_rag batch --document "宁波银行" --questions annual_report_questions.txt --output output/宁波银行_40题.md
```

每次问答只检索所选报告。对于未披露指标或证据不足的问题，应说明无法确定。无内嵌文字层的扫描版 PDF 会自动用 RapidOCR 补全（需安装 rapidocr-onnxruntime）；复杂表格中的数字建议对照原文复核。

详细说明见 [技术说明文档](技术说明文档.md)，新增报告的官方链接见 [资料来源](资料来源.md)，独立测试结果与已知问题见 [运行评估](运行评估.md)。
