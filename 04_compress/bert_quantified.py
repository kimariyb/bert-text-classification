import torch
from bert import MyBertConfig, MyBertModel


if __name__ == '__main__':
    config = MyBertConfig()
    model = MyBertModel(config).to(config.device)
    checkpoint = torch.load("./models/bert_original_20260530210633.pt", map_location=config.device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 量化 bert 模型
    print("量化模型...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8,
    )

    # 保存量化模型
    print("保存量化模型：", config.quantized_path)
    torch.save(quantized_model.state_dict(), config.quantized_path)