import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from dataset import load_dataset
from bert import MyBertConfig, MyBertModel
from transformers.utils import PaddingStrategy
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)


config = MyBertConfig()

model = MyBertModel(config).to(config.device)
checkpoint = torch.load("./models/bert_original_20260530204542.pt", map_location=config.device)
model.load_state_dict(checkpoint['model_state'])


def inference(test_dict: dict) -> dict:
    model.eval()

    test_tokens = config.tokenizer.batch_encode_plus(
        [test_dict['text']], padding=PaddingStrategy.MAX_LENGTH,
        truncation=True, max_length=config.pad_size, return_tensors='pt'
    )

    ids   = test_tokens["input_ids"].to(config.device)
    masks = test_tokens["attention_mask"].to(config.device)

    with torch.no_grad():
        out = model(ids, masks)
        pred_idx  = torch.argmax(out, dim=1).item()
        pred_label = config.class_list[pred_idx]

    test_dict['pred_class'] = pred_label

    if 'label' in test_dict:
        test_dict['label_class'] = config.class_list[test_dict['label']]
        test_dict['is_correct']  = pred_label == test_dict['label_class']

    return test_dict


def compute_metrics(pred_results: list[dict]):
    """
    从预测结果中提取真实标签与预测标签，计算分类指标。
    若结果中不含真实标签（纯推理场景），则跳过并返回 None。
    """
    if 'label_class' not in pred_results[0]:
        return None

    y_true = [r['label_class'] for r in pred_results]
    y_pred = [r['pred_class']  for r in pred_results]

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_macro":  f1_score(y_true, y_pred, average='macro',  zero_division=0),
        "f1_micro":  f1_score(y_true, y_pred, average='micro',  zero_division=0),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall":    recall_score(y_true, y_pred, average='macro',    zero_division=0),
    }

    # 每个类别的详细指标
    report = classification_report(
        y_true, y_pred,
        target_names=config.class_list,
        zero_division=0,
    )

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=config.class_list)

    return {"summary": metrics, "report": report, "confusion_matrix": cm}


def save_metrics(metrics: dict, save_dir: str = "./logs"):
    """将指标汇总保存为 CSV，逐类指标和混淆矩阵各存一个文件。"""
    # ── 1. 汇总指标 ─────────────────────────────────────────────
    summary_df = pd.DataFrame([metrics["summary"]])
    summary_path = f"{save_dir}/test_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"汇总指标已保存至 {summary_path}")

    # ── 2. 混淆矩阵 ──────────────────────────────────────────────
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"真实: {c}" for c in config.class_list],
        columns=[f"预测: {c}" for c in config.class_list],
    )
    cm_path = f"{save_dir}/test_confusion_matrix.csv"
    cm_df.to_csv(cm_path, encoding="utf-8")
    print(f"混淆矩阵已保存至 {cm_path}")


def print_metrics(metrics: dict):
    """在控制台格式化输出指标。"""
    s = metrics["summary"]
    print("\n" + "=" * 50)
    print(f"  Accuracy  : {s['accuracy']:.4f}")
    print(f"  F1 (macro): {s['f1_macro']:.4f}")
    print(f"  F1 (micro): {s['f1_micro']:.4f}")
    print(f"  Precision : {s['precision']:.4f}")
    print(f"  Recall    : {s['recall']:.4f}")
    print("=" * 50)
    print("\n── 各类别详细指标 ──")
    print(metrics["report"])
    print("── 混淆矩阵 ──")
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"真实: {c}" for c in config.class_list],
        columns=[f"预测: {c}" for c in config.class_list],
    )
    print(cm_df.to_string())
    print()


if __name__ == '__main__':
    # 加载测试数据
    test_data = load_dataset(config.test_path)
    test_data = [{"text": x[0], "label": x[1]} for x in test_data]

    # 逐条推理
    pred_results = []
    for test_dict in tqdm(test_data, desc="Predicting"):
        pred_results.append(inference(test_dict))

    # 保存预测结果 CSV
    df = pd.DataFrame(pred_results, columns=['text', 'label_class', 'pred_class', 'is_correct'])
    predict_path = "./logs/test_predict_result.csv"
    df.to_csv(predict_path, index=False, encoding="utf-8")
    print(f"预测结果已保存至 {predict_path}")

    # 计算并输出指标
    metrics = compute_metrics(pred_results)
    if metrics:
        print_metrics(metrics)
        save_metrics(metrics, save_dir="./logs")
    else:
        print("当前数据无真实标签，跳过指标计算。")