import os

import torch
import torch.nn as nn
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
        self.num_epoches = 3
        # 批次大小
        self.batch_size = 512
        # 填充长度
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # BERT 模型路径
        self.bert_path = "./pre_train"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path + '/bert_config.json')
        self.hidden_size = self.bert_config.hidden_size


class MyBertModel(nn.Module):
    def __init__(self, config):
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