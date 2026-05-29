import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertModel


class BERTConfig(object):
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


class BERTModel(nn.Module):
    def __init__(self, config):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask, return_dict=False)
        out = self.fc(pooled)
        return out


class TextCNNConfig(object):
    def __init__(self):
        self.model_name = "textCNN"
        self.data_path = "./data/"
        self.train_path = self.data_path + "train.txt"
        self.val_path = self.data_path + "val.txt"
        self.test_path = self.data_path + "test.txt"
        self.class_list = [x.strip() for x in open(
            self.data_path + "class.txt"
        ).readlines()]

        self.vocab_path = self.data_path + "vocab.pkl"

        self.save_path = "./model/"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += self.model_name + ".pt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.n_vocab = 0
        self.num_epoches = 30
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = 300
        self.filter_sizes = (2, 3, 4, 5)
        self.num_filters = 1024


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out