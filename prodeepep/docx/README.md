# Pro-DeepEP 异构时态图节点二分类流程

本目录提供在自建异构时态图上运行 DeepEP 的端到端实现，包含数据预处理、分层划分、欠采样训练、指标可视化与输出整理。所有路径、超参数均由 `config.py` 固定配置，无需命令行参数。

## 目录结构
- `config.py`：集中配置路径、随机种子与超参数。
- `preprocess.py`：读取 `Nodes.xlsx`、`Edges.xlsx`，生成 `protein_emb.npy`、`protein_label.npy`、`protein_matrix.npy`。
- `dataset.py`：分层划分 70/15/15，提供每轮欠采样接口。
- `metrics_report.py`：PPV、TPR、NPV、TNR、F1_Score、AUROC、AUPRC、ACC 计算与曲线绘制。
- `train.py`：复用 DeepEP CNN 训练、验证、测试流程，按示例日志格式输出。
- `main.py`：主入口，串联预处理与训练。
- `data/raw/`：放置 `Nodes.xlsx` 与 `Edges.xlsx`。
- `data/processed/`：保存生成的 `protein_emb.npy`、`protein_label.npy`、`protein_matrix.npy`。
- `output/`：训练阶段的模型权重、日志与可视化图片。
- `docx/`：项目文档（用户指南、技术报告、开发说明）。

## 使用步骤
1. 将 `Nodes.xlsx` 与 `Edges.xlsx` 置于 `prodeepep/data/raw/`。
2. 确认已安装 `tensorflow==1.0.0`、`networkx`、`pandas`、`openpyxl`、`node2vec`、`scikit-learn`、`matplotlib` 等依赖。
3. 在仓库根目录运行 `python prodeepep/main.py`，流程会自动预处理并开始训练。
4. 训练输出的模型、日志与图片将保存在 `prodeepep/output/`。

## 日志与评估
- 开始训练前打印图与数据划分统计，格式与示例一致。
- 每 10 个 epoch 输出一次验证集指标，训练完毕后输出测试集指标与 ROC/PR 图。
- 欠采样开关由 `config.ENABLE_UNDERSAMPLE` 控制，默认开启。