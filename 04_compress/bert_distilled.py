import logging
import os
import csv
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from bert import MyBertConfig, MyBertModel, DistillationLoss, BiLSTMModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import build_dataloader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

 
def setup_logger(log_dir: str = "logs", log_name: str = None) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if log_name is None:
        log_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_distill.log"
    log_path = os.path.join(log_dir, log_name)
 
    logger = logging.getLogger("Distiller")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
 
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler();          ch.setLevel(logging.INFO);  ch.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"日志保存至: {log_path}")
    return logger
 
 
class DistillCSVWriter:
    """
    train CSV: epoch, step, loss_total, loss_soft, loss_hard, acc, f1
    val   CSV: epoch, step, loss_total, loss_soft, loss_hard, acc, f1, precision, recall
    """
    TRAIN_FIELDS = ["epoch", "step", "loss_total", "loss_soft", "loss_hard", "acc", "f1"]
    VAL_FIELDS   = ["epoch", "step", "loss_total", "loss_soft", "loss_hard", "acc", "f1", "precision", "recall"]
 
    def __init__(self, log_dir: str, run_name: str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        tp = os.path.join(log_dir, f"{run_name}_distill_train.csv")
        vp = os.path.join(log_dir, f"{run_name}_distill_val.csv")
        self._tf = open(tp, "a", newline="", encoding="utf-8")
        self._vf = open(vp, "a", newline="", encoding="utf-8")
        self._tw = csv.DictWriter(self._tf, fieldnames=self.TRAIN_FIELDS)
        self._vw = csv.DictWriter(self._vf, fieldnames=self.VAL_FIELDS)
        if os.path.getsize(tp) == 0: self._tw.writeheader()
        if os.path.getsize(vp) == 0: self._vw.writeheader()
 
    def write_train(self, epoch, step, loss_comp: dict, metrics: dict):
        self._tw.writerow({
            "epoch": epoch + 1, "step": step,
            **{k: round(v, 6) for k, v in loss_comp.items()},
            "acc": round(metrics["acc"], 6), "f1": round(metrics["f1"], 6),
        })
        self._tf.flush()
 
    def write_val(self, epoch, step, loss_comp: dict, metrics: dict):
        self._vw.writerow({
            "epoch": epoch + 1, "step": step,
            **{k: round(v, 6) for k, v in loss_comp.items()},
            "acc":       round(metrics["acc"],       6),
            "f1":        round(metrics["f1"],        6),
            "precision": round(metrics["precision"], 6),
            "recall":    round(metrics["recall"],    6),
        })
        self._vf.flush()
 
    def close(self):
        self._tf.close(); self._vf.close()
 
 
def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "acc":       accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall":    recall_score(labels, preds, average="macro", zero_division=0),
    }
 

class Distiller:
    """
    知识蒸馏训练器。
 
    教师模型（BERT）固定权重，只做前向推理产生软标签；
    学生模型（BiLSTM）接收软标签 + 硬标签共同训练。
 
    训练/验证均在同一 LOG_INTERVAL-batch 窗口内统计，与 Trainer 保持一致。
    """
 
    LOG_INTERVAL = 100
 
    def __init__(
        self,
        teacher:    nn.Module,
        student:    nn.Module,
        optimizer:  torch.optim.Optimizer,
        config:     MyBertConfig,
        criterion:  DistillationLoss,
        scheduler=None,
        logger:     logging.Logger = None,
        log_dir:    str = "logs",
    ):
        self.teacher   = teacher
        self.student   = student
        self.optimizer = optimizer
        self.config    = config
        self.criterion = criterion
        self.scheduler = scheduler
        self.logger    = logger or logging.getLogger("Distiller")
 
        self.state = {"epoch": 0, "step": 0, "best_f1": 0.0}
 
        run_name        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_writer = DistillCSVWriter(log_dir=log_dir, run_name=run_name)
        self.save_path  = config.save_path.replace(".pt", "_student.pt")
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
 
    def distill(self, train_loader, val_loader, resume_path: str = None):
        if resume_path:
            self._load_checkpoint(resume_path)
 
        # 教师模型全程冻结，切换到 eval 模式（关闭 dropout / BN）
        self.teacher.to(self.config.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
 
        self.student.to(self.config.device)
 
        self.logger.info("=" * 60)
        self.logger.info("开始蒸馏训练")
        self.logger.info(f"  教师模型参数量: {sum(p.numel() for p in self.teacher.parameters()):,}")
        self.logger.info(f"  学生模型参数量: {sum(p.numel() for p in self.student.parameters()):,}")
        self.logger.info(f"  压缩比:         {sum(p.numel() for p in self.teacher.parameters()) / sum(p.numel() for p in self.student.parameters()):.1f}×")
        self.logger.info(f"  蒸馏温度 T:     {self.criterion.T}")
        self.logger.info(f"  软标签权重 α:   {self.criterion.alpha}")
        self.logger.info(f"  设备:           {self.config.device}")
        self.logger.info(f"  总轮数:         {self.config.num_epoches}")
        self.logger.info("=" * 60)
 
        try:
            for epoch in range(self.state["epoch"], self.config.num_epoches):
                self.state["epoch"] = epoch
                self.logger.info(f"\n{'─'*20} Epoch {epoch+1}/{self.config.num_epoches} {'─'*20}")
                self._train_one_epoch(train_loader, val_loader, epoch)
                if self.scheduler:
                    self.scheduler.step()
                    self.logger.debug(f"LR → {self.scheduler.get_last_lr()}")
            self.logger.info(f"蒸馏完成，学生模型最优 F1: {self.state['best_f1']:.4f}")
        finally:
            self.csv_writer.close()
 
    def _train_one_epoch(self, train_loader, val_loader, epoch: int):
        self.student.train()
 
        # 窗口缓冲
        win_loss_comp = {"loss_total": 0., "loss_soft": 0., "loss_hard": 0.}
        win_preds, win_labels = [], []
        val_iter    = iter(val_loader)
        num_batches = len(train_loader)
 
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Distilling")):
            loss, comp, preds, lbls = self._train_step(batch)
 
            for k in win_loss_comp:
                win_loss_comp[k] += comp[k]
            win_preds  = np.append(win_preds,  preds)
            win_labels = np.append(win_labels, lbls)
            self.state["step"] += 1
 
            is_log = (i + 1) % self.LOG_INTERVAL == 0 or i == num_batches - 1
            if not is_log:
                continue
 
            n = len(win_preds)
            avg_comp = {k: v / n for k, v in win_loss_comp.items()}
            metrics  = compute_metrics(win_labels, win_preds)
 
            self._log_train(epoch, i + 1, num_batches, avg_comp, metrics)
            self.csv_writer.write_train(epoch, i + 1, avg_comp, metrics)
 
            # 重置窗口
            win_loss_comp = {k: 0. for k in win_loss_comp}
            win_preds, win_labels = [], []
 
            # 验证窗口
            val_metrics, val_comp = self._evaluate_window(val_iter, val_loader)
            self._log_val(epoch, i + 1, val_comp, val_metrics)
            self.csv_writer.write_val(epoch, i + 1, val_comp, val_metrics)
            self._save_best(val_metrics["f1"])
 
            self.student.train()
 
    def _train_step(self, batch) -> tuple:
        input_ids, attention_mask, labels = batch
        input_ids      = input_ids.to(self.config.device)
        attention_mask = attention_mask.to(self.config.device)
        labels         = labels.to(self.config.device)
 
        # 教师前向（无梯度）
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids, attention_mask)
 
        # 学生前向
        self.optimizer.zero_grad()
        student_logits = self.student(input_ids, attention_mask)
        loss, comp = self.criterion(student_logits, teacher_logits, labels)
 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()
 
        preds = torch.argmax(student_logits, dim=1).cpu().numpy()
        return loss.item(), comp, preds, labels.cpu().numpy()
 
    @torch.no_grad()
    def _evaluate_window(self, val_iter, val_loader) -> tuple[dict, dict]:
        self.student.eval()
 
        win_loss_comp = {"loss_total": 0., "loss_soft": 0., "loss_hard": 0.}
        all_preds, all_labels = [], []
 
        for _ in range(self.LOG_INTERVAL):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch    = next(val_iter)
 
            input_ids, attention_mask, labels = batch
            input_ids      = input_ids.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)
            labels         = labels.to(self.config.device)
 
            teacher_logits = self.teacher(input_ids, attention_mask)
            student_logits = self.student(input_ids, attention_mask)
            _, comp        = self.criterion(student_logits, teacher_logits, labels)
 
            for k in win_loss_comp:
                win_loss_comp[k] += comp[k]
 
            all_preds  = np.append(all_preds,  torch.argmax(student_logits, dim=1).cpu().numpy())
            all_labels = np.append(all_labels, labels.cpu().numpy())
 
        n        = len(all_preds)
        avg_comp = {k: v / n for k, v in win_loss_comp.items()}
        metrics  = compute_metrics(all_labels, all_preds)
        return metrics, avg_comp
 
    def _log_train(self, epoch, step, total, comp: dict, metrics: dict):
        self.logger.info(
            f"[Train] Epoch {epoch+1}  Step {step:>4}/{total}  "
            f"Loss {comp['loss_total']:.4f} "
            f"(soft {comp['loss_soft']:.4f} hard {comp['loss_hard']:.4f})  "
            f"Acc {metrics['acc']:.2%}  F1 {metrics['f1']:.2%}"
        )
 
    def _log_val(self, epoch, step, comp: dict, metrics: dict):
        self.logger.info(
            f"[Val]   Epoch {epoch+1}  Step {step:>4}  "
            f"Loss {comp['loss_total']:.4f} "
            f"(soft {comp['loss_soft']:.4f} hard {comp['loss_hard']:.4f})  "
            f"Acc {metrics['acc']:.2%}  F1 {metrics['f1']:.2%}  "
            f"Precision {metrics['precision']:.2%}  Recall {metrics['recall']:.2%}"
        )
 
    def _save_best(self, val_f1: float):
        if val_f1 > self.state["best_f1"]:
            self.state["best_f1"] = val_f1
            torch.save({
                "model_state":   self.student.state_dict(),
                "trainer_state": self.state,
            }, self.save_path)
            self.logger.info(f"✓ 学生模型新最优 F1: {val_f1:.4f}，已保存至 {self.save_path}")
 
    def _load_checkpoint(self, path: str):
        if not os.path.exists(path):
            self.logger.warning(f"Checkpoint 不存在，忽略: {path}")
            return
        ckpt = torch.load(path, map_location=self.config.device)
        self.student.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "trainer_state" in ckpt:
            self.state = ckpt["trainer_state"]
        self.logger.info(f"从 {path} 恢复，Best F1: {self.state['best_f1']:.4f}")
 
 
if __name__ == "__main__":
    config  = MyBertConfig()
    log_dir = "logs"
    logger  = setup_logger(log_dir=log_dir)
 
    # 数据
    train_loader, val_loader = build_dataloader()
 
    # 教师模型（加载已训练好的 BERT）
    teacher = MyBertModel(config)
    ckpt = torch.load("./models/bert_original_20260530210633.pt", map_location=config.device)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()

    # 学生模型
    student   = BiLSTMModel(config)
    optimizer = AdamW(student.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epoches, eta_min=1e-6)

    # 蒸馏损失（温度 T=2，软标签权重 α=0.7）
    criterion = DistillationLoss(temperature=2.0, alpha=0.7)
 
    distiller = Distiller(
        teacher=teacher, student=student,
        optimizer=optimizer, config=config,
        criterion=criterion, scheduler=scheduler,
        logger=logger, log_dir=log_dir,
    )
 
    distiller.distill(train_loader, val_loader)