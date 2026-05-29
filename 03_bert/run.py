from utils import build_dataset, build_iterator
from train import train, test
from models import Config, Model
import numpy as np
import torch


if __name__ == '__main__':

    config = Config()

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    train_data, val_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    val_iter = build_iterator(val_data, config)
    test_iter = build_iterator(test_data, config)

    model = Model(config).to(config.device)

    train(config, model, train_iter, val_iter)
    test(config, model, test_iter)


