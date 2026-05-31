import os
import csv
import logging
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from bert import MyBertConfig, MyBertModel
from dataset import build_dataloader


def setup_logger(log_dir: str = "logs", log_name: str = None) -> logging.Logger:
    """
    配置 logger，同时输出到控制台和日志文件。

    Args:
        log_dir:  日志文件夹路径
        log_name: 日志文件名（默认按时间戳自动生成）

    Returns:
        配置好的 logger 实例
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if log_name is None:
        log_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"

    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.DEBUG)

    # 防止重复添加 handler（在 Jupyter / 多次调用场景下常见）
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # 文件 handler（DEBUG 级别，保留完整信息）
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"日志文件保存至: {log_path}")
    return logger


class MetricsCSVWriter:
    """
    将训练 / 验证指标追加写入 CSV 文件。
    每次调用 write() 时立即 flush，保证进程中断后数据不丢失。

    train CSV 列: epoch, step, loss, acc, f1
    val   CSV 列: epoch, step, loss, acc, f1, precision, recall
    """

    TRAIN_FIELDS = ["epoch", "step", "loss", "acc", "f1"]
    VAL_FIELDS   = ["epoch", "step", "loss", "acc", "f1", "precision", "recall"]

    def __init__(self, log_dir: str, run_name: str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        train_path = os.path.join(log_dir, f"{run_name}_train.csv")
        val_path   = os.path.join(log_dir, f"{run_name}_val.csv")

        # 以追加模式打开，支持断点续训后继续写入
        self._train_f  = open(train_path, "a", newline="", encoding="utf-8")
        self._val_f    = open(val_path,   "a", newline="", encoding="utf-8")
        self._train_w  = csv.DictWriter(self._train_f, fieldnames=self.TRAIN_FIELDS)
        self._val_w    = csv.DictWriter(self._val_f,   fieldnames=self.VAL_FIELDS)

        # 仅在新文件时写表头
        if os.path.getsize(train_path) == 0:
            self._train_w.writeheader()
        if os.path.getsize(val_path) == 0:
            self._val_w.writeheader()

    def write_train(self, epoch: int, step: int, loss: float, metrics: dict):
        self._train_w.writerow({
            "epoch": epoch + 1,
            "step":  step,
            "loss":  round(loss, 6),
            "acc":   round(metrics["acc"], 6),
            "f1":    round(metrics["f1"],  6),
        })
        self._train_f.flush()

    def write_val(self, epoch: int, step: int, loss: float, metrics: dict):
        self._val_w.writerow({
            "epoch":     epoch + 1,
            "step":      step,
            "loss":      round(loss,                  6),
            "acc":       round(metrics["acc"],        6),
            "f1":        round(metrics["f1"],         6),
            "precision": round(metrics["precision"],  6),
            "recall":    round(metrics["recall"],     6),
        })
        self._val_f.flush()

    def close(self):
        self._train_f.close()
        self._val_f.close()


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """计算分类任务常用指标，返回 dict。"""
    return {
        "acc":       accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall":    recall_score(labels, preds, average="macro", zero_division=0),
    }


def loss_fn(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(pred, label)


class Trainer:
    """
    封装完整训练流程，包括：
      - 训练 / 验证均在同一 100-batch 窗口内统计 loss 与指标
      - 日志记录（logging）+ 指标持久化（CSV）
      - 学习率调度（LR Scheduler）
      - 梯度裁剪（Gradient Clipping）
      - 最优模型保存
      - 断点续训（Resume from Checkpoint）
    """

    # 每隔多少个 batch 统计并输出一次
    LOG_INTERVAL = 100

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: MyBertConfig,
        scheduler=None,
        logger: logging.Logger = None,
        log_dir: str = "logs",
    ):
        self.model      = model
        self.optimizer  = optimizer
        self.config     = config
        self.scheduler  = scheduler
        self.logger     = logger or logging.getLogger("Trainer")

        # 训练状态，便于断点续训时恢复
        self.state = {
            "epoch":   0,
            "step":    0,
            "best_f1": 0.0,
        }

        # CSV 记录器，run_name 用时间戳区分不同训练
        run_name       = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_writer = MetricsCSVWriter(log_dir=log_dir, run_name=run_name)

        Path(config.save_path).parent.mkdir(parents=True, exist_ok=True)

    def train(self, train_loader, val_loader, resume_path: str = None):
        """
        主训练入口。

        Args:
            train_loader: 训练集 DataLoader
            val_loader:   验证集 DataLoader
            resume_path:  checkpoint 路径；传入则从该 checkpoint 继续训练
        """
        if resume_path:
            self._load_checkpoint(resume_path)

        self.logger.info("=" * 60)
        self.logger.info("开始训练")
        self.logger.info(f"  设备:        {self.config.device}")
        self.logger.info(f"  总轮数:      {self.config.num_epoches}")
        self.logger.info(f"  批次大小:    {train_loader.batch_size}")
        self.logger.info(f"  学习率:      {self.config.learning_rate}")
        self.logger.info(f"  保存路径:    {self.config.save_path}")
        self.logger.info(f"  Log 间隔:    每 {self.LOG_INTERVAL} 个 batch")
        self.logger.info("=" * 60)

        self.model.to(self.config.device)

        try:
            start_epoch = self.state["epoch"]
            for epoch in range(start_epoch, self.config.num_epoches):
                self.state["epoch"] = epoch
                self.logger.info(f"\n{'─'*20} Epoch {epoch + 1}/{self.config.num_epoches} {'─'*20}")

                self._train_one_epoch(train_loader, val_loader, epoch)

                if self.scheduler is not None:
                    self.scheduler.step()
                    self.logger.debug(f"LR 调整为: {self.scheduler.get_last_lr()}")

            self.logger.info("训练完成，最优 F1: {:.4f}".format(self.state["best_f1"]))
        finally:
            # 保证异常退出时 CSV 文件也正常关闭
            self.csv_writer.close()

    def _train_one_epoch(self, train_loader, val_loader, epoch: int):
        self.model.train()

        # 窗口缓冲区（每 LOG_INTERVAL 个 batch 重置一次）
        win_loss   = 0.0
        win_preds, win_labels = [], []

        # 验证窗口缓冲区（与训练同步，在触发点评估最近 LOG_INTERVAL 个 val batch）
        val_iter    = iter(val_loader)
        num_batches = len(train_loader)

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")):
            loss, pred_list, label_list = self._train_step(batch)

            win_loss  += loss
            win_preds  = np.append(win_preds,  pred_list)
            win_labels = np.append(win_labels, label_list)
            self.state["step"] += 1

            # ── 每 LOG_INTERVAL 步（或最后一步）同时输出 train + val ──
            is_log_step = (i + 1) % self.LOG_INTERVAL == 0 or i == num_batches - 1
            if not is_log_step:
                continue

            step_tag = i + 1

            # ── Train 指标 ────────────────────────────────────────────
            train_metrics = compute_metrics(win_labels, win_preds)
            train_loss    = win_loss / len(win_preds)   # 每样本平均 loss

            self._log_train(epoch, step_tag, num_batches, train_loss, train_metrics)
            self.csv_writer.write_train(epoch, step_tag, train_loss, train_metrics)

            # 重置训练窗口
            win_loss  = 0.0
            win_preds, win_labels = [], []

            # ── Val 指标（采样最近 LOG_INTERVAL 个 val batch）────────
            val_metrics, val_loss = self._evaluate_window(val_iter, val_loader)

            self._log_val(epoch, step_tag, val_loss, val_metrics)
            self.csv_writer.write_val(epoch, step_tag, val_loss, val_metrics)
            self._save_best(val_metrics["f1"])

            # 恢复训练模式
            self.model.train()

    def _train_step(self, batch) -> tuple:
        """单步前向 + 反向传播，返回 (loss_value, pred_array, label_array)。"""
        input_ids, attention_mask, labels = batch
        input_ids      = input_ids.to(self.config.device)
        attention_mask = attention_mask.to(self.config.device)
        labels         = labels.to(self.config.device)

        self.optimizer.zero_grad()
        logits = self.model(input_ids, attention_mask)
        loss   = loss_fn(logits, labels)
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        preds  = torch.argmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        return loss.item(), preds, labels

    @torch.no_grad()
    def _evaluate_window(self, val_iter, val_loader) -> tuple[dict, float]:
        """
        从 val_iter 中消费最多 LOG_INTERVAL 个 batch 进行评估。
        val_iter 耗尽后自动重置，保证每轮都能取到数据。

        Returns:
            (metrics_dict, avg_loss_per_sample)
        """
        self.model.eval()

        win_loss  = 0.0
        all_preds, all_labels = [], []

        for _ in range(self.LOG_INTERVAL):
            try:
                batch = next(val_iter)
            except StopIteration:
                # val_loader 耗尽，重置迭代器
                val_iter = iter(val_loader)
                batch    = next(val_iter)

            input_ids, attention_mask, labels = batch
            input_ids      = input_ids.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)
            labels         = labels.to(self.config.device)

            logits    = self.model(input_ids, attention_mask)
            win_loss += loss_fn(logits, labels).item()

            pred_list  = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds  = np.append(all_preds,  pred_list)
            all_labels = np.append(all_labels, labels.cpu().numpy())

        avg_loss = win_loss / len(all_preds)
        metrics  = compute_metrics(all_labels, all_preds)
        return metrics, avg_loss

    def _log_train(self, epoch: int, step: int, total: int, loss: float, metrics: dict):
        self.logger.info(
            f"[Train] Epoch {epoch + 1}  Step {step:>4}/{total}  "
            f"Loss {loss:.4f}  Acc {metrics['acc']:.2%}  F1 {metrics['f1']:.2%}"
        )

    def _log_val(self, epoch: int, step: int, loss: float, metrics: dict):
        self.logger.info(
            f"[Val]   Epoch {epoch + 1}  Step {step:>4}  "
            f"Loss {loss:.4f}  Acc {metrics['acc']:.2%}  F1 {metrics['f1']:.2%}  "
            f"Precision {metrics['precision']:.2%}  Recall {metrics['recall']:.2%}"
        )

    def _save_best(self, val_f1: float):
        """若 val_f1 刷新最优，则保存完整 checkpoint。"""
        if val_f1 > self.state["best_f1"]:
            self.state["best_f1"] = val_f1
            self._save_checkpoint(self.config.save_path)
            self.logger.info(f"✓ 新最优 F1: {val_f1:.4f}，模型已保存至 {self.config.save_path}")

    def _save_checkpoint(self, path: str):
        """保存模型权重、优化器状态和训练状态（支持断点续训）。"""
        checkpoint = {
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "trainer_state":   self.state,
        }
        torch.save(checkpoint, path)

    def _load_checkpoint(self, path: str):
        """从 checkpoint 恢复训练状态。"""
        if not os.path.exists(path):
            self.logger.warning(f"Checkpoint 不存在，忽略: {path}")
            return

        self.logger.info(f"从 checkpoint 恢复: {path}")
        checkpoint = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if self.scheduler and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        self.state = checkpoint["trainer_state"]
        self.logger.info(
            f"已恢复至 Epoch {self.state['epoch']}，"
            f"Step {self.state['step']}，Best F1 {self.state['best_f1']:.4f}"
        )

if __name__ == "__main__":
    config = MyBertConfig()

    log_dir = "logs"

    # 日志
    logger = setup_logger(log_dir=log_dir)

    # 数据
    train_loader, val_loader = build_dataloader()

    # 模型 & 优化器
    model = MyBertModel(config)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    # 学习率调度器（可替换为其他 Scheduler）
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epoches, eta_min=1e-6)

    # 训练
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        scheduler=scheduler,
        logger=logger,
        log_dir=log_dir,
    )

    # 如需断点续训，传入 checkpoint 路径：
    # trainer.train(train_loader, val_loader, resume_path="checkpoints/best.pt")
    trainer.train(train_loader, val_loader)