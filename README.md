# BERT 文本分类项目

基于机器学习与深度学习的中文新闻分类项目，实现了从传统机器学习到预训练语言模型的完整文本分类流程，并进一步探索模型压缩技术，包括动态量化、结构化剪枝以及知识蒸馏。

## 项目简介

本项目使用今日头条新闻分类数据集（Toutiao News Dataset）完成中文新闻文本分类任务。

实现了以下模型：

- Random Forest（Baseline）
- FastText
- BERT
- BERT Dynamic Quantization (INT8)
- BERT Structured Pruning (30%)
- BERT → BiLSTM Knowledge Distillation

并对模型的分类性能与压缩效果进行了系统评估。

## 文本分类类别

| 类别 | 说明 |
|------|------|
| finance | 金融 |
| realty | 房产 |
| stocks | 股票 |
| education | 教育 |
| science | 科技 |
| society | 社会 |
| politics | 政治 |
| sports | 体育 |
| game | 游戏 |
| entertainment | 娱乐 |

## 技术路线

```
                 新闻文本
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼

   TF-IDF+RF     FastText      BERT

                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼

          INT8量化          30%剪枝         知识蒸馏(BiLSTM)
```

## 模型介绍

### 1. Random Forest（Baseline）

作为传统机器学习基线模型：

```text
文本
 ↓
TF-IDF
 ↓
Random Forest
 ↓
分类结果
```

特点：

* 训练速度快
* 可解释性较好
* 作为 Baseline 模型


### 2. FastText

采用 Facebook 提出的 FastText 文本分类模型。

特点：

* 训练效率高
* 支持 n-gram 特征
* 对中文短文本分类效果优秀

```text
Text
 ↓
Embedding
 ↓
Average Pooling
 ↓
Softmax
```


### 3. BERT

采用预训练语言模型 BERT 进行微调。

```text
Text
 ↓
Tokenizer
 ↓
BERT Encoder
 ↓
CLS Token
 ↓
Linear Classifier
 ↓
Prediction
```

特点：

* 利用上下文语义信息
* 显著优于传统方法
* 获得最佳分类性能


## 模型压缩

为了提升模型部署效率，对 BERT 进行了三种压缩方案。

### 1. INT8 动态量化

将权重从 FP32 压缩为 INT8：

```text
FP32 → INT8
```

优点：

* 模型体积更小
* 推理速度提升
* 精度损失较小

### 2. 30% 结构化剪枝

对低重要性参数进行剪枝：

```text
Original Parameters
        ↓
   Remove 30%
        ↓
 Pruned Model
```

优点：

* 减少参数量
* 降低计算开销


### 3. 知识蒸馏

教师模型：

```text
BERT
```

学生模型：

```text
BiLSTM
```

蒸馏流程：

```text
Teacher(BERT)
      ↓
 Soft Labels
      ↓
 Student(BiLSTM)
```

目标：

* 保留 BERT 的知识
* 获得轻量级部署模型

# 实验结果

## 基础模型对比

| Model         |   Accuracy |  Precision |     Recall |         F1 |
| ------------- | ---------: | ---------: | ---------: | ---------: |
| Random Forest |     0.7520 |     0.7822 |     0.7520 |     0.7595 |
| FastText      |     0.9091 |     0.9091 |     0.9091 |     0.9091 |
| BERT          | **0.9412** | **0.9415** | **0.9412** | **0.9413** |

### 性能提升

| 对比              | Accuracy 提升 |
| --------------- | ----------: |
| RF → FastText   |     +15.71% |
| FastText → BERT |      +3.21% |
| RF → BERT       |     +18.92% |

可以看到：

* FastText 显著优于传统机器学习方法；
* BERT 进一步利用上下文语义信息获得最佳性能；
* Random Forest 作为 Baseline 与深度学习模型存在明显差距。

---

## BERT 压缩实验

| Model            |   Accuracy |   F1-Macro |  Precision |     Recall | F1 Retention |
| ---------------- | ---------: | ---------: | ---------: | ---------: | -----------: |
| Original BERT    | **0.9440** | **0.9441** | **0.9443** | **0.9440** |         100% |
| INT8 Quantized   |     0.9249 |     0.9250 |     0.9266 |     0.9249 |        98.0% |
| 30% Pruned       |     0.9324 |     0.9320 |     0.9327 |     0.9324 |        98.7% |
| Distilled BiLSTM |     0.8367 |     0.8370 |     0.8381 |     0.8367 |        88.7% |

---

## 压缩效果分析

### INT8量化

```text
Accuracy:
94.40% → 92.49%
```

仅损失：

```text
1.91%
```

保留：

```text
98.0%
```

说明动态量化对分类性能影响较小，适合部署场景。

---

### 30%剪枝

```text
Accuracy:
94.40% → 93.24%
```

仅下降：

```text
1.16%
```

保留：

```text
98.7%
```

说明 BERT 存在一定参数冗余。

---

### 知识蒸馏

```text
Accuracy:
94.40% → 83.67%
```

保留：

```text
88.7%
```

虽然性能下降较明显，但学生模型推理速度和部署成本更低。


## 环境配置

```bash
Python >= 3.10

torch
transformers
scikit-learn
fasttext
numpy
pandas
tqdm
```

