import os
import torch
import pickle as pkl
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


UNK, PAD, CLS = '[UNK]', '[PAD]', '[CLS]'
MAX_VOCAB_SIZE = 10000


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if not line:
                continue

            content = line.split("\t")[0]
            for word in tokenizer(content):
                vocab_dict[word] = vocab_dict.get(word, 0) + 1

    # Move vocabulary construction outside the loop to avoid re-computing on every line
    vocab_list = sorted([_ for _ in vocab_dict.items() if _[1] >= min_freq],
                        key=lambda x: x[1], reverse=True)[:max_size]

    vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dict.update({UNK: len(vocab_dict), PAD: len(vocab_dict) + 1, CLS: len(vocab_dict) + 2})

    return vocab_dict


def build_dataset(config):
    def load_dataset(path, pad_size: int = 32):
        contents = []
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                if not line:
                    continue

                content, label = line.split("\t")
                token = config.tokenizer.tokenize(content)
                token = ['CLS'] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size

                contents.append((token_ids, int(label), seq_len, mask))

        return contents

    train = load_dataset(config.train_path, config.pad_size)
    val = load_dataset(config.val_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return train, val, test


def build_dataset_CNN(config):
    tokenizer = lambda x: [y for y in x]

    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))

    print("Vocab size:", len(vocab))

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                if not line:
                    continue

                content, label = line.split("\t")
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)

                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size

                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                contents.append((words_line, int(label), seq_len))

        return contents

    train = load_dataset(config.train_path, config.pad_size)
    val = load_dataset(config.val_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return vocab, train, val, test

class TextDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __getitem__(self, index):
        return self.batches[index]

    def __len__(self):
        return len(self.batches)


def collate_fn(batch_data, model_name, device):
    x = torch.LongTensor([item[0] for item in batch_data]).to(device)
    y = torch.LongTensor([item[1] for item in batch_data]).to(device)
    seq_len = torch.LongTensor([item[2] for item in batch_data]).to(device)

    if model_name == "bert":
        mask = torch.LongTensor([item[3] for item in batch_data]).to(device)
        return (x, seq_len, mask), y
    else:
        return (x, seq_len), y

def build_iterator(dataset, config):
    text_dataset = TextDataset(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: collate_fn(x, config.model_name, config.device)
    )

    return dataloader