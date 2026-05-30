import pandas as pd
import jieba
import os


def preprocess(path: str, class_path: str, output_path: str = None) -> pd.DataFrame:
    if not os.path.exists(path) or not os.path.exists(class_path):
        raise FileNotFoundError

    # create id_to_label dict
    id_to_label = {}

    # Load class data
    class_df = pd.read_csv(class_path, header=None, names=["class"])
    for i, class_name in enumerate(class_df['class']):
        id_to_label[i] = str(class_name).strip()

    # Load data
    df = pd.read_csv(path, sep="\t", header=None, names=["sentence", "label"])

    # drop na
    df.dropna()

    df["text_length"] = df['sentence'].apply(lambda x: len(str(x)))

    mean_text_length = df["text_length"].mean()
    std_text_length = df["text_length"].std()

    print(f"Sentence mean: {mean_text_length}")
    print(f"Sentence std: {std_text_length}")

    # split word and transform to fasttext style
    df['label_name'] = df['label'].apply(lambda x: '__label__' + id_to_label[int(x)])
    df['words'] = df['sentence'].apply(lambda x: ' '.join(jieba.lcut(str(x))))
    df = df[['label_name', 'words']]

    if output_path:
        df.to_csv(output_path, sep="\t", index=False, header=False)
        print(f"Save to {output_path}")

    return df


if __name__ == "__main__":
    DATA_ROOT = "./data/"
    class_path = os.path.join(DATA_ROOT, "class.txt")
    train_path = os.path.join(DATA_ROOT, "train.txt")
    val_path = os.path.join(DATA_ROOT, "val.txt")
    test_path = os.path.join(DATA_ROOT, "test.txt")

    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError

    if os.path.exists(train_path):
        preprocess(train_path, class_path, os.path.join(DATA_ROOT, "train_processed.csv"))
    else:
        print("No train data")

    if os.path.exists(val_path):
        preprocess(val_path, class_path, os.path.join(DATA_ROOT, "val_processed.csv"))
    else:
        print("No val data")

    if os.path.exists(test_path):
        preprocess(test_path, class_path, os.path.join(DATA_ROOT, "test_processed.csv"))
    else:
        print("No test data")