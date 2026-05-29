import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim import AdamW
from tqdm import tqdm


def loss_fn(pred, label):
    return F.cross_entropy(pred, label)


criterion = nn.KLDivLoss()

def loss_fn_kd(pred, label, teacher_pred, T=2, alpha=0.8):
    student_pred = F.log_softmax(pred / T, dim=1)
    teacher_pred = F.softmax(teacher_pred / T, dim=1)

    soft_loss = criterion(student_pred, teacher_pred)
    hard_loss = loss_fn(pred, label)

    kd_loss = soft_loss * alpha * T * T + hard_loss * (1 - alpha)
    return kd_loss


def fetch_teacher_pred(teacher_model, train_iter):
    teacher_model.eval()
    teacher_pred = []

    with torch.no_grad():
        for i, (data_batch, label_batch) in enumerate(tqdm(train_iter)):
            pred = teacher_model(data_batch)
            teacher_pred.append(pred)

    return teacher_pred


def train(config, model, train_iter, val_iter, test_iter):
    model.train()
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

    for epoch in range(config.num_epoches):
        total_batch = 0
        print("Epoch:", epoch + 1)
        for i, (trains, labels) in enumerate(tqdm(train_iter)):
            model.zero_grad()
            pred = model(trains)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            total_batch += 1

            if total_batch % 400 == 0 and total_batch != 0:
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

    test(config, model, test_iter)


def train_kd(cnn_config, bert_model, cnn_model, bert_train_iter, cnn_train_iter, cnn_val_iter, cnn_test_iter):
    param_optimizer = list(cnn_model.named_parameters())
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

    optimizer = AdamW(optimizer_groups, lr=cnn_config.learning_rate, weight_decay=0.01)
    val_best_loss = float("inf")
    cnn_model.train()
    bert_model.eval()
    teacher_pred = fetch_teacher_pred(bert_model, bert_train_iter)
    for epoch in range(cnn_config.num_epoches):
        total_batch = 0
        print("Epoch:", epoch + 1)
        for i, (trains, labels) in enumerate(tqdm(cnn_train_iter)):
            cnn_model.zero_grad()
            pred = cnn_model(trains)
            loss = loss_fn_kd(pred, labels, teacher_pred[i])
            loss.backward()
            optimizer.step()
            total_batch += 1
            if total_batch % 400 == 0 and total_batch != 0:
                label = labels.data.cpu()
                pred = torch.max(pred.data, dim=1).indices.cpu()
                train_acc = accuracy_score(label, pred)
                val_acc, val_loss = evaluate(cnn_config, cnn_model, cnn_val_iter)
                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    torch.save(cnn_model.state_dict(), cnn_config.save_path)

                message = "Iter: {0:>6},  Train loss: {1:>5.2},  Train acc: {2:>6.2%},  Val loss: {3:>5.2}, Val acc: {4:>6.2%}"
                print(message.format(total_batch, loss.item(), train_acc, val_loss, val_acc))
                cnn_model.train()

    test(cnn_config, cnn_model, cnn_test_iter)


def evaluate(config, model, data_iter, test=False):
    model.eval()
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
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)

    message = "Test loss: {0:>5.2},  Test acc: {1:>6.2%}"
    print(message.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score ...")
    print(test_report)
    print("Confusion Matrix ...")
    print(test_confusion)