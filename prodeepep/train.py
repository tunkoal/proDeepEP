# -*- coding: utf-8 -*-
"""
训练、验证与测试流程，直接复用 DeepEP 的 CNN 主体。
"""
import os
import numpy as np
import tensorflow as tf
from DeepEP.protein_cnn import CnnModel

import config
from dataset import ProteinDataset
from metrics_report import compute_metrics, format_epoch_log, plot_val_history, plot_test_curves, print_metrics_report


# 固定随机种子，确保可复现
np.random.seed(config.RANDOM_SEED)
tf.set_random_seed(config.RANDOM_SEED)


def _format_count(value, total):
    percent = (value / total) * 100 if total else 0
    return f"{value:6d} ({percent:6.2f}%)"


def print_graph_report(graph, dataset):
    stats = dataset.stats()
    total_nodes = graph.number_of_nodes()
    total_labels = stats['total']
    pos = stats['pos']
    neg = stats['neg']
    edge_num = graph.number_of_edges()

    print('Building graph...')
    print('========== Graph Report =========\n')
    print(f"[Nodes] total = {total_nodes:,}")
    print(f"\n[Labels ] total = {total_labels:,}")
    print(f"  - positive : {_format_count(pos, total_labels)}")
    print(f"  - negative : {_format_count(neg, total_labels)}\n")
    print(f"[Edges] total = {edge_num:,}\n")
    print("[Splits] size and ratio over labeled nodes")
    for split_name, key in [('train', 'train'), ('val  ', 'val'), ('test ', 'test')]:
        split = stats[key]
        total = split['total']
        pos_cnt = split['pos']
        neg_cnt = split['neg']
        print(
            f"  - {split_name}: {total:5d} ({(total / total_labels) * 100:6.2f}%) | [+] {pos_cnt:5d} ({(pos_cnt / total) * 100:6.2f}%)  [-] {neg_cnt:5d} ({(neg_cnt / total) * 100:6.2f}%)"
        )
    print('==================================')


def _run_inference(sess, model, x_data, e_data, y_data):
    step_size = 300
    probs = []
    losses = []
    for i in range(0, len(y_data), step_size):
        batch_x = x_data[i:i + step_size]
        batch_e = e_data[i:i + step_size]
        batch_y = y_data[i:i + step_size]
        loss_val, prob = sess.run(
            [model.loss_cnn, model.logits_pred],
            feed_dict={model.x: batch_x, model.e: batch_e, model.y: batch_y, model.dropout_keep_prob: 1.0},
        )
        losses.append(loss_val)
        probs.append(prob)
    return float(np.mean(losses)), np.concatenate(probs, axis=0)


def train_and_evaluate(graph):
    dataset = ProteinDataset()
    print_graph_report(graph, dataset)

    train_pos = dataset.stats()['train']['pos']
    if config.ENABLE_UNDERSAMPLE:
        decay_steps = (train_pos * 2) / config.BATCH_SIZE
    else:
        decay_steps = dataset.stats()['train']['total'] / config.BATCH_SIZE

    model = CnnModel(config.INIT_LEARNING_RATE, decay_steps, config.DECAY_RATE)

    history = []
    saver = tf.train.Saver()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Start training...')

        for epoch in range(1, config.EPOCH_NUM_CNN + 1):
            train_x, train_e, train_y = dataset.epoch_training_data(enable_undersample=config.ENABLE_UNDERSAMPLE)
            epoch_losses = []
            for idx in range(0, len(train_y), config.BATCH_SIZE):
                batch_x = train_x[idx:idx + config.BATCH_SIZE]
                batch_e = train_e[idx:idx + config.BATCH_SIZE]
                batch_y = train_y[idx:idx + config.BATCH_SIZE]
                loss_val, _ = sess.run(
                    [model.loss_cnn, model.optimizer_cnn],
                    feed_dict={model.x: batch_x, model.e: batch_e, model.y: batch_y, model.dropout_keep_prob: config.KEEP_PRO},
                )
                epoch_losses.append(loss_val)

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

            if epoch % config.LOG_INTERVAL == 0 or epoch == 1:
                val_x, val_e, val_y = dataset.val_data
                val_loss, val_prob = _run_inference(sess, model, val_x, val_e, val_y)
                val_metrics = compute_metrics(val_y, val_prob)
                history.append({'epoch': epoch, 'metrics': val_metrics})
                print(format_epoch_log(epoch, val_loss, val_metrics))

        # 保存模型参数
        saver.save(sess, os.path.join(config.OUTPUT_DIR, 'deepep_cnn.ckpt'))

        # 验证集曲线
        plot_val_history(history, config.OUTPUT_DIR)

        # 测试集评估
        test_x, test_e, test_y = dataset.test_data
        test_loss, test_prob = _run_inference(sess, model, test_x, test_e, test_y)
        test_metrics = compute_metrics(test_y, test_prob)
        print('\n==== Test set performance ====')
        print_metrics_report('', test_metrics)

        plot_test_curves(test_y.reshape(-1), test_prob.reshape(-1), config.OUTPUT_DIR)

        return {
            'val_history': history,
            'test_metrics': test_metrics,
            'test_loss': test_loss,
        }
