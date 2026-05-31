import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from dataset import load_dataset
from bert import MyBertConfig, MyBertModel, BiLSTMModel
from transformers.utils import PaddingStrategy
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

config  = MyBertConfig()
LOG_DIR = "./logs/evaluate"

# ---------------------------------------------------------------------------
# 模型注册表：name → (model_class, checkpoint_path, is_quantized)
# is_quantized=True  → 文件是量化后的裸 state_dict，加载前需先重建量化结构
# is_quantized=False → checkpoint 格式 {"model_state": ...} 或普通 state_dict
# 按需注释掉不想测的模型
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "original":  (MyBertModel, "./models/bert_original_20260530210633.pt",        False),
    "quantized": (MyBertModel, "./models/bert_quantized_20260530212028.pt",        True),
    "pruned":    (MyBertModel, "./models/bert_pruned_20260530212046.pt",           False),
    "distilled": (BiLSTMModel, "./models/bert_original_20260530214842_student.pt", False),
}


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_model(
    model_class,
    checkpoint_path: str,
    is_quantized: bool = False,
) -> tuple[torch.nn.Module, torch.device]:
    """
    统一加载入口，自动处理三种存储格式：

    1. checkpoint 格式  {"model_state": state_dict}
       → 原始模型、蒸馏模型、剪枝模型
    2. 量化模型裸 state_dict（键名含 _packed_params）
       → 必须先对原始模型做 quantize_dynamic，再 load_state_dict
    3. 整个模型对象（torch.save(model, path)）
       → 直接返回，无需实例化
    """
    # 量化模型必须在 CPU 加载和推理
    device = torch.device("cpu") if is_quantized else config.device
    ckpt   = torch.load(checkpoint_path, map_location=device)
    
    # ── 情况 3：整个模型对象 ────────────────────────────────────
    if not isinstance(ckpt, dict):
        model = ckpt

    # ── 情况 1：checkpoint 格式 ─────────────────────────────────
    elif "model_state" in ckpt:
        model = model_class(config)
        model.load_state_dict(ckpt["model_state"])

    # ── 情况 2：量化模型裸 state_dict ───────────────────────────
    elif is_quantized:
        model = model_class(config)
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        model.load_state_dict(ckpt)

    else:
        model = model_class(config)
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# 单条推理
# ---------------------------------------------------------------------------

def inference(model: torch.nn.Module, test_dict: dict, device: torch.device) -> dict:
    test_tokens = config.tokenizer.batch_encode_plus(
        [test_dict["text"]], padding=PaddingStrategy.MAX_LENGTH,
        truncation=True, max_length=config.pad_size, return_tensors="pt",
    )
    ids   = test_tokens["input_ids"].to(device)
    masks = test_tokens["attention_mask"].to(device)

    with torch.no_grad():
        out        = model(ids, masks)
        pred_idx   = torch.argmax(out, dim=1).item()
        pred_label = config.class_list[pred_idx]

    result = {**test_dict, "pred_class": pred_label}
    if "label" in test_dict:
        result["label_class"] = config.class_list[test_dict["label"]]
        result["is_correct"]  = pred_label == result["label_class"]
    return result


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def compute_metrics(pred_results: list[dict]):
    if "label_class" not in pred_results[0]:
        return None

    y_true = [r["label_class"] for r in pred_results]
    y_pred = [r["pred_class"]  for r in pred_results]

    return {
        "summary": {
            "accuracy":  accuracy_score(y_true, y_pred),
            "f1_macro":  f1_score(y_true, y_pred, average="macro",  zero_division=0),
            "f1_micro":  f1_score(y_true, y_pred, average="micro",  zero_division=0),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall":    recall_score(y_true, y_pred, average="macro",    zero_division=0),
        },
        "report": classification_report(
            y_true, y_pred, target_names=config.class_list, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=config.class_list),
    }


# ---------------------------------------------------------------------------
# 单模型评测
# ---------------------------------------------------------------------------

def evaluate_one(
    model_name:  str,
    model_class,
    ckpt_path:   str,
    is_quantized: bool,
    test_data:   list[dict],
):
    print(f"\n{'─'*50}")
    print(f"  评测: {model_name}  ({ckpt_path})")
    print(f"{'─'*50}")

    model, device = load_model(model_class, ckpt_path, is_quantized)
    results = [
        inference(model, d, device)
        for d in tqdm(test_data, desc=f"{model_name}")
    ]

    # 保存逐条预测结果
    save_dir = Path(LOG_DIR) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cols = ["text", "label_class", "pred_class", "is_correct"]
    pd.DataFrame(results, columns=cols).to_csv(
        save_dir / "predictions.csv", index=False, encoding="utf-8"
    )

    metrics = compute_metrics(results)
    if metrics is None:
        return None

    # 保存混淆矩阵
    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index  =[f"真实: {c}" for c in config.class_list],
        columns=[f"预测: {c}" for c in config.class_list],
    )
    cm_df.to_csv(save_dir / "confusion_matrix.csv", encoding="utf-8")

    # 打印
    s = metrics["summary"]
    print(f"  Accuracy : {s['accuracy']:.4f}")
    print(f"  F1 macro : {s['f1_macro']:.4f}")
    print(f"  F1 micro : {s['f1_micro']:.4f}")
    print(f"  Precision: {s['precision']:.4f}")
    print(f"  Recall   : {s['recall']:.4f}")
    print(f"\n── 各类别详细指标 ──")
    print(metrics["report"])

    # 释放显存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics["summary"]


# ---------------------------------------------------------------------------
# 横向对比表
# ---------------------------------------------------------------------------

def print_comparison(all_summaries: dict[str, dict]):
    """输出所有模型的横向对比表，标注相对原始模型的 F1 保留率。"""
    rows = [{"model": name, **s} for name, s in all_summaries.items()]
    df   = pd.DataFrame(rows).set_index("model")

    # F1 保留率（相对原始模型）
    if "original" in all_summaries:
        base_f1 = all_summaries["original"]["f1_macro"]
        df["f1_retention"] = (df["f1_macro"] / base_f1).map("{:.1%}".format)

    # 格式化数值列
    for col in ["accuracy", "f1_macro", "f1_micro", "precision", "recall"]:
        df[col] = df[col].map("{:.4f}".format)

    print(f"\n{'='*60}")
    print("  模型横向对比")
    print(f"{'='*60}")
    print(df.to_string())
    print(f"{'='*60}\n")

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{LOG_DIR}/comparison.csv", encoding="utf-8")
    print(f"对比表已保存至 {LOG_DIR}/comparison.csv")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_data = load_dataset(config.test_path)
    test_data = [{"text": x[0], "label": x[1]} for x in test_data]
    print(f"测试集大小: {len(test_data)} 条")

    all_summaries = {}
    for name, (model_class, ckpt_path, is_quantized) in MODEL_REGISTRY.items():
        summary = evaluate_one(name, model_class, ckpt_path, is_quantized, test_data)
        if summary:
            all_summaries[name] = summary

    if len(all_summaries) > 1:
        print_comparison(all_summaries)