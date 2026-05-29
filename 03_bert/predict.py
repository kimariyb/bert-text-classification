import numpy as np
import torch

from models import Config, Model


id_to_name = {
    0: 'finance',
    1: 'realty',
    2: 'stocks',
    3: 'education',
    4: 'science',
    5: 'society',
    6: 'politics',
    7: 'sports',
    8: 'game',
    9: 'entertainment'
}


def inference(model, config, input_text, pad_size=32):
    content = config.tokenizer.tokenize(input_text)
    content = ['CLS'] + content
    seq_len = len(content)

    token_ids = config.tokenizer.convert_tokens_to_ids(content)
    if pad_size:
        if len(content) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token_ids))
            token_ids += ([0] * (pad_size - len(token_ids)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size

        x = torch.LongTensor(token_ids).to(config.device)
        seq_len = torch.LongTensor([seq_len]).to(config.device)
        mask = torch.LongTensor(mask).to(config.device)
        x = x.unsqueeze(0)
        seq_len = seq_len.unsqueeze(0)
        mask = mask.unsqueeze(0)
        data = (x, seq_len, mask)
        pred = model(data)
        pred = torch.max(pred.data, dim=1).indices.cpu().numpy()

        return pred

    return None


if __name__ == '__main__':
    config = Config()

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    model = Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))

    input_text = "体育比赛非常精彩，运动员表现优异"
    pred = inference(model, config, input_text)
    print(id_to_name[pred.item()])