import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim import AdamW


def loss_fn(pred, label):
    return F.cross_entropy(pred, label)


def train(config, model, train_iter, val_iter):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_groups = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_groups, lr=config.learning_rate, weight_decay=0.01)
    val_best_loss = float("inf")

    model.train()
    for epoch in range(config.num_epoches):
        total_batch = 0
        print("Epoch:", epoch + 1)
        for i, (trains, labels) in enumerate(tqdm(train_iter)):
            pred = model(trains) # forward
            model.zero_grad()
            loss = loss_fn(pred, labels)
            loss.backward() #  backward
            optimizer.step()
            if total_batch % 100 == 0 and total_batch != 0:
                label = labels.data.cpu()
                pred = torch.max(pred.data, dim=1).indices.cpu()
                train_acc = accuracy_score(label, pred)
                val_acc, val_loss = evaluate(config, model, val_iter)

                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    torch.save(model.state_dict(), config.save_path)

                message = "Iter: {0:>6},  Train loss: {1:>5.2},  Train acc: {2:>6.2%},  Val loss: {3:>5.2}, Val acc: {4:>6.2%}"
                print(message.format(total_batch, loss.item(), train_acc, val_loss, val_acc))
                model.train()

            total_batch += 1


def evaluate(config, model, data_iter, test=False):
    loss_total = 0.0
    preds_all = np.array([], dtype=int)
    labels_all  = np.array([], dtype=int)

    with torch.no_grad():
        for text, labels in data_iter:
            pred = model(text)
            loss = loss_fn(pred, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            pred = torch.max(pred.data, dim=1).indices.cpu().numpy()

            preds_all = np.append(preds_all, pred)
            labels_all = np.append(labels_all, labels)

    acc = accuracy_score(labels_all, preds_all)

    if test:
        report = classification_report(labels_all, preds_all,
                                       target_names=config.class_list, digits=4)
        confusion = confusion_matrix(labels_all, preds_all)
        return acc, loss_total / len(data_iter), report, confusion
    else:
        return acc, loss_total / len(data_iter)


def test(config, model, test_iter):
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)

    message = "Test loss: {0:>5.2},  Test acc: {1:>6.2%}"
    print(message.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score ...")
    print(test_report)
    print("Confusion Matrix ...")
    print(test_confusion)