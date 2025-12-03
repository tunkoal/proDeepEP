# -*- coding: utf-8 -*-
"""
数据预处理模块：
1) 读取 Nodes.xlsx 与 Edges.xlsx；
2) 构建 PPI 图并执行 node2vec 生成蛋白质嵌入；
3) 生成符合 DeepEP 输入格式的 protein_emb.npy、protein_label.npy、protein_matrix.npy。
"""
import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

import config


# 固定随机种子，确保 node2vec 与采样一致
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


def _validate_columns(frame, required_cols):
    """严格校验列名，防止静默出错。"""
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise KeyError(f"缺少必需列: {missing}")


def load_raw_tables():
    """读取原始 Excel 表，保持文件顺序。"""
    nodes_df = pd.read_excel(config.NODES_FILE, engine="openpyxl")
    edges_df = pd.read_excel(config.EDGES_FILE, engine="openpyxl")
    _validate_columns(nodes_df, ['Systematic_Name', 'Is_Essential'] + [f't{i}' for i in range(36)])
    _validate_columns(edges_df, ['Source', 'Target'])
    return nodes_df, edges_df


def build_graph(nodes_df, edges_df):
    """根据节点与边表构建无向图，确保所有节点被添加。"""
    graph = nx.Graph()
    node_ids = nodes_df['Systematic_Name'].astype(str).tolist()
    graph.add_nodes_from(node_ids)
    edge_pairs = list(zip(edges_df['Source'].astype(str), edges_df['Target'].astype(str)))
    graph.add_edges_from(edge_pairs)
    return graph, node_ids


def generate_node2vec_embeddings(graph, node_ids):
    """按照原论文参数生成 64 维 node2vec 嵌入。"""
    n2v = Node2Vec(
        graph,
        dimensions=config.EMBEDDING_SIZE,
        walk_length=20,
        num_walks=10,
        p=1,
        q=1,
        workers=4,
        seed=config.RANDOM_SEED,
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([model.wv[str(node)] for node in node_ids], dtype=np.float32)
    return embeddings


def reshape_expression(nodes_df):
    """将 36 个时间片重塑为 (3, 12) 的基因表达张量。"""
    time_cols = [f't{i}' for i in range(36)]
    expression = nodes_df[time_cols].to_numpy(dtype=np.float32)
    expression = expression.reshape(-1, config.CHANNEL_SIZE, config.TIME_STEPS)
    return expression


def extract_labels(nodes_df):
    """提取标签并保持二维形状 (N, 1)。"""
    labels = nodes_df[['Is_Essential']].to_numpy(dtype=np.int32)
    return labels


def ensure_dirs():
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def save_node_index(node_ids):
    with open(config.NODE_INDEX_FILE, 'w', encoding='utf-8') as fw:
        fw.write('\n'.join(node_ids))


def generate_all():
    """执行全流程预处理。"""
    ensure_dirs()
    nodes_df, edges_df = load_raw_tables()
    graph, node_ids = build_graph(nodes_df, edges_df)

    protein_emb = generate_node2vec_embeddings(graph, node_ids)
    protein_matrix = reshape_expression(nodes_df)
    protein_label = extract_labels(nodes_df)

    np.save(config.PROTEIN_EMB_FILE, protein_emb)
    np.save(config.PROTEIN_MATRIX_FILE, protein_matrix)
    np.save(config.PROTEIN_LABEL_FILE, protein_label)
    save_node_index(node_ids)
    return {
        'graph': graph,
        'node_ids': node_ids,
        'protein_emb': protein_emb,
        'protein_matrix': protein_matrix,
        'protein_label': protein_label,
    }


if __name__ == '__main__':
    generate_all()