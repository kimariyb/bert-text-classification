import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertConfig, BertModel
from datetime import datetime


class MyBertConfig:
    def __init__(self):
        # 模型名称
        self.model_name = "bert"
        # 数据路径
        self.data_path = "/home/kimariyb/project/toutiao/data/"
        # 训练集路径
        self.train_path = self.data_path + "train.txt"
        # 验证集路径
        self.val_path = self.data_path + "val.txt"
        # 测试集路径
        self.test_path = self.data_path + "test.txt"
        # 类别列表
        self.class_list = [x.strip() for x in open(
            self.data_path + "class.txt"
        ).readlines()]

        # 模型保存路径
        self.save_path = "./models/"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += self.model_name + "_original_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".pt"

        # 量化模型保存路径
        self.quantized_path = "./models/"
        if not os.path.exists(self.quantized_path):
            os.mkdir(self.quantized_path)
        self.quantized_path += self.model_name + "_quantized_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".pt"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 类别数
        self.num_classes = len(self.class_list)
        # 训练轮数
        self.num_epoches = 10
        # 批次大小
        self.batch_size = 256
        # 填充长度
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # BERT 模型路径
        self.bert_path = "./pre_train"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path + '/bert_config.json')
        self.hidden_size = self.bert_config.hidden_size

        # 蒸馏 biLSTM 模型参数配置
        self.embed_size = 256
        self.hidden_size_lstm = 512
        self.num_layers = 6
        self.bilstm_save_model_path = self.data_path + "bilstm_classifer_model.pt"
        self.dropout = 0.3


class MyBertModel(nn.Module):
    def __init__(self, config: MyBertConfig):
        super(MyBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path=config.bert_path, 
            config=config.bert_config
        )
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        out = self.fc(pooled)
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, config: MyBertConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.bert_config.vocab_size, config.embed_size)
        self.lstm = nn.LSTM(
            config.embed_size, config.hidden_size_lstm, config.num_layers,
            bidirectional=True, batch_first=True, dropout=config.dropout if config.num_layers > 1 else 0,
        )
        self.attn = nn.Linear(config.hidden_size_lstm * 2, 1)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size_lstm * 2, config.num_classes)
 
    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
 
        # 过滤 [CLS](101) 和 [SEP](102) 的掩码
        cls_sep_mask = (input_ids != 101) & (input_ids != 102)
        valid_mask = attention_mask.bool() & cls_sep_mask        # [B, L]
 
        x = x * valid_mask.unsqueeze(-1)                        # [B, L, E]
        lstm_out, _ = self.lstm(x)                              # [B, L, H*2]
 
        # 注意力权重：padding 位置填 -inf 使 softmax 后归零
        attn_scores = self.attn(lstm_out).squeeze(-1)        # [B, L]
        attn_scores = attn_scores.masked_fill(~valid_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)        # [B, L]
 
        # 加权求和
        pooled = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, H*2]
        pooled = self.dropout(pooled)

        return self.fc(pooled)
 

class DistillationLoss(nn.Module):
    """
    知识蒸馏损失 = α * 软标签损失 + (1-α) * 硬标签损失
 
    软标签损失：KL 散度，度量学生与教师在温度 T 下的输出分布差异。
    硬标签损失：标准交叉熵，保证学生仍然拟合真实标签。
 
    温度 T 越高，教师的 softmax 输出越平滑，类间相对关系（暗知识）越
    明显，学生越容易学习；T=1 退化为普通交叉熵。
 
    Args:
        temperature: 蒸馏温度 T，通常取 2~8，默认 4
        alpha:       软标签损失权重，(1-alpha) 为硬标签权重，默认 0.7
    """
 
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
 
    def forward(
        self,
        student_logits: torch.Tensor,   # [B, C]
        teacher_logits: torch.Tensor,   # [B, C]
        labels:         torch.Tensor,   # [B]
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: 加权总损失（标量）
            components: 各项 loss 的数值 dict，用于日志
        """
        # 软标签损失（KL 散度，乘以 T² 是标准做法，抵消梯度缩放）
        soft_student = F.log_softmax(student_logits / self.T, dim=-1)
        soft_teacher = F.softmax(teacher_logits    / self.T, dim=-1)
        loss_soft    = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.T ** 2)
 
        # 硬标签损失
        loss_hard = F.cross_entropy(student_logits, labels)
 
        total = self.alpha * loss_soft + (1.0 - self.alpha) * loss_hard
 
        return total, {
            "loss_total": total.item(),
            "loss_soft":  loss_soft.item(),
            "loss_hard":  loss_hard.item(),
        }
 