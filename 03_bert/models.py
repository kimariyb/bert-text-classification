import os

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel


class Config(object):
    def __init__(self):
        self.model_name = "bert"
        self.data_path = "./data/"
        self.train_path = self.data_path + "train.txt"
        self.val_path = self.data_path + "val.txt"
        self.test_path = self.data_path + "test.txt"
        self.class_list = [x.strip() for x in open(
            self.data_path + "class.txt"
        ).readlines()]

        self.save_path = "./model/"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += self.model_name + ".pt"

        self.quantized_path = "./model"
        if not os.path.exists(self.quantized_path):
            os.mkdir(self.quantized_path)
        self.quantized_path += self.model_name + "_quantized_" + ".pt"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_classes = len(self.class_list)
        self.num_epoches = 2
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-5
        self.bert_path = "./pre_train"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path + '/bert_config.json')
        self.hidden_size = self.bert_config.hidden_size


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask, return_dict=False)
        out = self.fc(pooled)
        return out