import torch

ckpt = torch.load("./models/bert_quantized_20260530212028.pt", map_location="cpu")
print(type(ckpt))
