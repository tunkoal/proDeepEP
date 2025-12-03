# -*- coding: utf-8 -*-
"""
数据集划分与欠采样策略实现，保持 DeepEP 输入格式。
"""
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import config


class ProteinDataset:
    """封装数据加载、分层划分与训练欠采样。"""

    def __init__(self):
        self.rng = np.random.RandomState(config.RANDOM_SEED)
        self.expression = np.load(config.PROTEIN_MATRIX_FILE)
        self.embedding = np.load(config.PROTEIN_EMB_FILE)
        self.labels = np.load(config.PROTEIN_LABEL_FILE)
        self._validate_shapes()
        self._stratified_split()

    def _validate_shapes(self):
        if self.expression.shape[0] != self.embedding.shape[0] or self.embedding.shape[0] != self.labels.shape[0]:
            raise ValueError('特征与标签的样本数量不一致')

    def _stratified_split(self):
        y_flat = self.labels.reshape(-1)
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=config.TRAIN_RATIO, random_state=config.RANDOM_SEED)
        train_idx, temp_idx = next(splitter.split(np.zeros_like(y_flat), y_flat))
        temp_y = y_flat[temp_idx]
        val_ratio = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        splitter_val = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio, random_state=config.RANDOM_SEED)
        val_idx_rel, test_idx_rel = next(splitter_val.split(np.zeros_like(temp_y), temp_y))
        val_idx = temp_idx[val_idx_rel]
        test_idx = temp_idx[test_idx_rel]
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

    def _subset(self, indices):
        return (
            self.expression[indices],
            self.embedding[indices],
            self.labels[indices],
        )

    @property
    def train_data(self):
        return self._subset(self.train_idx)

    @property
    def val_data(self):
        return self._subset(self.val_idx)

    @property
    def test_data(self):
        return self._subset(self.test_idx)

    def epoch_training_data(self, enable_undersample=True):
        """
        每轮训练采样：保留全部正样本，从负样本中等量抽取。
        """
        x_train, e_train, y_train = self.train_data
        y_flat = y_train.reshape(-1)
        pos_mask = y_flat == 1
        neg_mask = y_flat == 0
        pos_idx = np.where(pos_mask)[0]
        neg_idx = np.where(neg_mask)[0]

        if enable_undersample:
            if len(neg_idx) < len(pos_idx):
                raise ValueError('负样本数量不足以完成欠采样，请检查数据分布。')
            selected_neg = self.rng.choice(neg_idx, size=len(pos_idx), replace=False)
            selected = np.concatenate([pos_idx, selected_neg])
        else:
            selected = np.arange(len(y_flat))

        self.rng.shuffle(selected)
        return x_train[selected], e_train[selected], y_train[selected]

    def stats(self):
        total = len(self.labels)
        y_flat = self.labels.reshape(-1)
        pos = int(np.sum(y_flat == 1))
        neg = int(np.sum(y_flat == 0))
        return {
            'total': total,
            'pos': pos,
            'neg': neg,
            'train': self._split_stats(self.train_idx),
            'val': self._split_stats(self.val_idx),
            'test': self._split_stats(self.test_idx),
        }

    def _split_stats(self, indices):
        y = self.labels[indices].reshape(-1)
        total = len(y)
        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))
        return {'total': total, 'pos': pos, 'neg': neg}

    def edge_count(self, graph):
        return graph.number_of_edges()