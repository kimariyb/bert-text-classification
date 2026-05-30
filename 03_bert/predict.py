import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from dataset import load_dataset
from bert import MyBertConfig, MyBertModel
from transformers.utils import PaddingStrategy


config = MyBertConfig()

model = MyBertModel(config).to(config.device)
checkpoint = torch.load(config.save_path, map_location=config.device)
model.load_state_dict(checkpoint['model_state']) 


def inference(test_dict: dict[str]):
    model.eval()

    test_contents = test_dict['text']

    # 获取 ids 和 mask
    test_tokens = config.tokenizer.batch_encode_plus(
        [test_contents], padding=PaddingStrategy.MAX_LENGTH, 
        truncation=True, max_length=config.pad_size, return_tensors='pt'
    )

    ids = test_tokens["input_ids"].to(config.device)
    masks = test_tokens["attention_mask"].to(config.device)

    with torch.no_grad():
        out = model(ids, masks)
        pred_idx = torch.argmax(out, dim=1).item()
        pred_label = config.class_list[pred_idx]

    test_dict['pred_class'] = pred_label
    
    if 'label' in test_dict:
        test_dict['label_class'] = config.class_list[test_dict['label']]
        test_dict['is_correct'] = pred_label == config.class_list[test_dict['label']]

    return test_dict


if __name__ == '__main__':
    # 加载测试数据
    test_data = load_dataset(config.test_path)
    
    # 将测试数据改写为 dict
    test_data = [{"text": x[0], "label": x[1]} for x in test_data]

    pred_results = []
    for test_dict in tqdm(test_data, desc="Predicting"):
        pred_results.append(inference(test_dict))

    # 保存 CSV
    df = pd.DataFrame(pred_results, columns=['text', 'label_class', 'pred_class', 'is_correct'])
    
    # 关键：保存路径不能和模型权重同一个！
    df.to_csv("./logs/test_predict_result.csv", index=False, encoding="utf-8")
    print("预测完成，已保存至 test_predict_result.csv")