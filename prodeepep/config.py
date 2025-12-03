# -*- coding: utf-8 -*-
"""
全局配置：路径、随机种子与超参数集中在此文件顶部，避免使用命令行参数。
"""
import os

# 随机种子
RANDOM_SEED = 42

# 数据与输出目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 文件名约定
NODES_FILE = os.path.join(RAW_DATA_DIR, 'Nodes.xlsx')
EDGES_FILE = os.path.join(RAW_DATA_DIR, 'Edges.xlsx')
PROTEIN_EMB_FILE = os.path.join(PROCESSED_DATA_DIR, 'protein_emb.npy')
PROTEIN_LABEL_FILE = os.path.join(PROCESSED_DATA_DIR, 'protein_label.npy')
PROTEIN_MATRIX_FILE = os.path.join(PROCESSED_DATA_DIR, 'protein_matrix.npy')
NODE_INDEX_FILE = os.path.join(PROCESSED_DATA_DIR, 'node_index.txt')

# 训练相关超参数（保持 DeepEP 原设定）
BATCH_SIZE = 32
EPOCH_NUM_CNN = 40
KEEP_PRO = 0.95
INIT_LEARNING_RATE = 0.001
DECAY_RATE = 0.96
TIME_STEPS = 12
CHANNEL_SIZE = 3
EMBEDDING_SIZE = 64
EMBEDDING_FN_SIZE = 312

# 数据划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 欠采样开关
ENABLE_UNDERSAMPLE = False

# 节点评估打印间隔
LOG_INTERVAL = 10

# 评估与绘图参数
PLOT_STYLE = 'seaborn-colorblind'
PLOT_DPI = 200