import pandas as pd
import numpy as np
import jieba
import os

from typing import Optional


def preprocess(path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError

    # Load data
    df = pd.read_csv(path, sep="\t", header=None, 
        names=["sentence", "label"], encoding="utf-8")

    # drop na
    df.dropna(inplace=True)

    df["text_length"] = df['sentence'].apply(lambda x: len(str(x)))

    mean_text_length = df["text_length"].mean()
    std_text_length = df["text_length"].std()

    print(f"Sentence mean: {mean_text_length}")
    print(f"Sentence std: {std_text_length}")

    # split word
    df['words'] = df['sentence'].apply(lambda x: jieba.lcut(str(x)))

    if output_path:
        df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
        print(f"Save to {output_path}")

    return df


if __name__ == "__main__":
    DATA_ROOT = "./data/"
    train_path = os.path.join(DATA_ROOT, "train.txt")
    val_path = os.path.join(DATA_ROOT, "val.txt")
    test_path = os.path.join(DATA_ROOT, "test.txt")

    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError

    if os.path.exists(train_path):
        preprocess(train_path, os.path.join(DATA_ROOT, "train_processed.csv"))
    else:
        print("No train data")

    if os.path.exists(val_path):
        preprocess(val_path, os.path.join(DATA_ROOT, "val_processed.csv"))
    else:
        print("No val data")

    if os.path.exists(test_path):
        preprocess(test_path, os.path.join(DATA_ROOT, "test_processed.csv"))
    else:
        print("No test data")