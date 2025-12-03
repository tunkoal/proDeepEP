# -*- coding: utf-8 -*-
"""
指标计算与可视化模块。
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics

import config

plt.style.use(config.PLOT_STYLE)


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0.0


def compute_metrics(y_true, y_prob):
    y_true = y_true.reshape(-1)
    y_prob = y_prob.reshape(-1)
    y_pred = (y_prob > 0.5).astype(int)

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ppv = _safe_div(tp, tp + fp)
    tpr = _safe_div(tp, tp + fn)
    npv = _safe_div(tn, tn + fn)
    tnr = _safe_div(tn, tn + fp)
    f1_score = metrics.f1_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)

    try:
        auroc = metrics.roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.0
    try:
        auprc = metrics.average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = 0.0

    return {
        'PPV': ppv,
        'TPR': tpr,
        'NPV': npv,
        'TNR': tnr,
        'F1_Score': f1_score,
        'AUROC': auroc,
        'AUPRC': auprc,
        'ACC': acc,
    }


def format_epoch_log(epoch, loss, metrics_dict):
    """按照示例格式输出每个 epoch 的指标。"""
    return (
        f"Epoch {epoch:03d} | loss {loss:.4f} | "
        f"PPV {metrics_dict['PPV']:.4f} TPR {metrics_dict['TPR']:.4f} | "
        f"NPV {metrics_dict['NPV']:.4f} TNR {metrics_dict['TNR']:.4f} | "
        f"F1_Score {metrics_dict['F1_Score']:.4f} "
        f"AUROC {metrics_dict['AUROC']:.4f} AUPRC {metrics_dict['AUPRC']:.4f} "
        f"ACC {metrics_dict['ACC']:.4f}"
    )


def plot_val_history(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = [item['epoch'] for item in history]
    for metric_name in ['PPV', 'NPV', 'TPR', 'TNR', 'F1_Score', 'AUROC', 'AUPRC', 'ACC']:
        values = [item['metrics'][metric_name] for item in history]
        plt.figure(dpi=config.PLOT_DPI)
        plt.plot(epochs, values, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'Validation {metric_name}')
        plt.grid(True)
        file_name = f"val_{metric_name}.png"
        plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
        plt.close()


def plot_test_curves(y_true, y_prob, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)

    plt.figure(dpi=config.PLOT_DPI)
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'test_roc_curve.png'), bbox_inches='tight')
    plt.close()

    plt.figure(dpi=config.PLOT_DPI)
    plt.plot(recall, precision, label='Precision-Recall')
    plt.xlabel('Recall (TPR)')
    plt.ylabel('Precision (PPV)')
    plt.title('Test Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'test_ppv_tpr_curve.png'), bbox_inches='tight')
    plt.close()


def print_metrics_report(title, metrics_dict):
    print(title)
    for k, v in metrics_dict.items():
        print(f"{k:<8}: {v:.4f}")