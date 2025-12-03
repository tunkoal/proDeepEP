# -*- coding: utf-8 -*-
"""
主入口：串联数据预处理与模型训练，所有路径与超参数由 config 统一管理。
"""
import os
import tensorflow as tf

import config
import preprocess
from train import train_and_evaluate


def main():
    # 关闭冗余日志
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.ERROR)

    preprocess_outputs = preprocess.generate_all()
    graph = preprocess_outputs['graph']
    train_and_evaluate(graph)


if __name__ == '__main__':
    main()