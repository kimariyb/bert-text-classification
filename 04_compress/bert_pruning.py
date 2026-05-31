import torch
import torch.nn.utils.prune as prune
from bert import MyBertModel, MyBertConfig
from datetime import datetime


def compute_sparsity(model: torch.nn.Module) -> float:
    """
    计算模型整体稀疏度（零参数占比）。
    使用 p.ne(0).sum() 逐元素判断，而非 p.abs().sum() > 0（后者只要
    tensor 中有任意一个非零值就算整个 tensor 非零，结果严重偏低）。
    """
    total = sum(p.numel() for p in model.parameters())
    nonzero  = sum(p.ne(0).sum().item() for p in model.parameters())
    return 1.0 - nonzero / total
 

def get_prunable_params(
    model: torch.nn.Module,
    types: tuple = (torch.nn.Linear,),
) -> list[tuple]:
    """
    收集模型中所有可剪枝的 (module, param_name) 元组。
 
    Args:
        model: 待剪枝的模型
        types: 需要剪枝的层类型，默认只剪 Linear（BERT 的核心层）
 
    Returns:
        [(module, 'weight'), ...]  格式，可直接传给 global_unstructured
    """
    params_to_prune = []
    for module in model.modules():
        if isinstance(module, types):
            params_to_prune.append((module, "weight"))
    return params_to_prune


def remove_pruning_masks(model: torch.nn.Module):
    """
    将剪枝掩码永久化：把 weight_orig * weight_mask → weight，
    并移除中间参数（weight_orig / weight_mask）。
 
    若不调用此函数，保存的 state_dict 中会含有 weight_orig 和
    weight_mask 两个额外键，加载时需要同样带有剪枝 hook 的模型。
    调用后 state_dict 与原始模型结构完全一致，可直接加载。
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except ValueError:
                # 该层未挂载剪枝 hook，跳过
                pass


if __name__ == "__main__":
    # 加载模型
    config = MyBertConfig()
    model = MyBertModel(config)
    checkpoint = torch.load("./models/bert_original_20260530210633.pt", map_location=config.device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(config.device)
    model.eval()
 
    amount = 0.3  # 剪枝比例（0~1）
    save_path = f"./models/bert_pruned_{datetime.now().strftime('%Y%m%d%H%M%S')}.pt"
 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{'─'*50}")
    print(f"剪枝前")
    print(f"  参数总量:  {total_params:,}")
    print(f"  稀疏度:    {compute_sparsity(model):.2%}")
    print(f"{'─'*50}")

    # ── 全局非结构化剪枝 ──────────────────────────────────────────────────
    # global_unstructured 要求：
    #   1. params_to_prune: List[(module, param_name_str)]  ← 元组列表，非 Tensor 列表
    #   2. pruning_method:  prune.L1Unstructured / RandomUnstructured 等
    #   3. amount:          剪枝比例（0~1）
    params_to_prune = get_prunable_params(model)
    print(f"待剪枝层数:  {len(params_to_prune)}")
    print(f"剪枝比例:    {amount:.0%}")
 
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,   # 按 L1 范数剪掉绝对值最小的权重
        amount=amount,
    )

    # ── 剪枝后（掩码生效，尚未永久化）────────────────────────────────────
    print(f"{'─'*50}")
    print(f"剪枝后（掩码已挂载）")
    print(f"  稀疏度:    {compute_sparsity(model):.2%}")

    # ── 永久化掩码 ────────────────────────────────────────────────────────
    # 将 weight_orig * weight_mask 合并回 weight，
    # 移除 weight_orig / weight_mask，state_dict 结构恢复正常
    remove_pruning_masks(model)
 
    print(f"{'─'*50}")
    print(f"剪枝后（掩码已永久化）")
    print(f"  参数总量:  {sum(p.numel() for p in model.parameters()):,}  ← 数量不变，零值已固化")
    print(f"  稀疏度:    {compute_sparsity(model):.2%}")
    print(f"{'─'*50}")
 
    # ── 保存 ──────────────────────────────────────────────────────────────
    torch.save({"model_state": model.state_dict()}, save_path)
    print(f"剪枝模型已保存至: {save_path}")


"""
──────────────────────────────────────────────────
剪枝前
  参数总量:  102,275,338
  稀疏度:    0.00%
──────────────────────────────────────────────────
待剪枝层数:  74
剪枝比例:    30%
──────────────────────────────────────────────────
剪枝后（掩码已挂载）
  稀疏度:    0.00%
──────────────────────────────────────────────────
剪枝后（掩码已永久化）
  参数总量:  102,275,338  ← 数量不变，零值已固化
  稀疏度:    25.09%
──────────────────────────────────────────────────
"""