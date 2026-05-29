import torch
from tqdm import tqdm

from bert.models import Config
from torch.utils.data import Dataset, DataLoader


def build_dataset(config: Config):
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