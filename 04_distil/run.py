from utils import build_dataset, build_dataset_CNN, build_iterator
from train import train, train_kd
from models import BERTConfig, BERTModel, TextCNNConfig, TextCNN
import numpy as np
import torch


TRAIN_TASK = "KD"
cnn_config = TextCNNConfig()
bert_config = BERTConfig()


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    if TRAIN_TASK == "BERT":

        train_data, val_data, test_data = build_dataset(bert_config)
        train_iter = build_iterator(train_data, bert_config)
        val_iter = build_iterator(val_data, bert_config)
        test_iter = build_iterator(test_data, bert_config)
        bert_model = BERTModel(bert_config).to(bert_config.device)

        train(bert_config, bert_model, train_iter, val_iter, test_iter)

    elif TRAIN_TASK == "KD":
        bert_train_data, _, _  = build_dataset(bert_config)
        bert_train_iter = build_iterator(bert_train_data, bert_config)

        vocab, cnn_train_data, cnn_val_data, cnn_test_data = build_dataset_CNN(cnn_config)
        cnn_train_iter = build_iterator(cnn_train_data, cnn_config)
        cnn_val_iter = build_iterator(cnn_val_data, cnn_config)
        cnn_test_iter = build_iterator(cnn_test_data, cnn_config)
        cnn_config.n_vocab = len(vocab)

        bert_model = BERTModel(bert_config).to(bert_config.device)
        cnn_model = TextCNN(cnn_config).to(cnn_config.device)

        train_kd(cnn_config, bert_model, cnn_model, bert_train_iter,  cnn_train_iter, cnn_val_iter, cnn_test_iter)






