import torch
from tqdm import tqdm

from bert import MyBertConfig
from transformers.utils import PaddingStrategy
from torch.utils.data import Dataset, DataLoader


config = MyBertConfig()


class TextDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __getitem__(self, index):
        x = self.batches[index][0]
        y = self.batches[index][1]

        return x, y

    def __len__(self):
        return len(self.batches)


def load_dataset(path: str) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if not line:
                continue

            content, label = line.split("\t")
            data.append((content, int(label)))
        
    return data

def build_dataloader():
    train_data = load_dataset(config.train_path)
    val_data = load_dataset(config.val_path)

    # 创建数据集
    train_dataset = TextDataset(train_data)
    val_dataset = TextDataset(val_data)

    # 创建数据加载器
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader


def collate_fn(batch_data):
    x, y = zip(*batch_data)

    tokens = config.tokenizer.batch_encode_plus(
        x, padding=PaddingStrategy.MAX_LENGTH, 
        max_length=config.pad_size, 
        truncation=True,
        return_tensors="pt"
    )

    ids, masks = tokens["input_ids"], tokens["attention_mask"]
    
    # 转换为张量
    y = torch.tensor(y)
    
    return ids, masks, y