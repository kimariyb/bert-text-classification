# BERT 文本分类项目

今日头条新闻文本分类项目，基于多阶段模型实现从基准模型到 BERT 模型压缩的完整流程。

## 项目结构

```
toutiao/
├── 01_baseline/          # 基线模型（TF-IDF + 机器学习）
├── 02_fasttext/          # FastText 模型
├── 03_bert/               # BERT 模型
├── 04_compress/           # BERT 模型压缩
├── data/                  # 原始数据
└── README.md
```

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

## 模型阶段

### 01_baseline - 基线模型

基于 TF-IDF 特征提取，结合 RandomForest 和 XGBoost 构建基准分类模型。

**技术栈**：sklearn、TfidfVectorizer、XGBoost

**流程**：数据预处理 → TF-IDF向量化 → 模型训练 → 评估

### 02_fasttext - FastText 模型

使用 jieba 分词进行中文分词，基于 FastText 实现文本分类。

**技术栈**：fasttext、jieba

**特点**：支持 n-gram 特征，自动调参

### 03_bert - BERT 模型

基于预训练 BERT 的中文文本分类模型，支持分布式训练和 API 服务部署。

**技术栈**：PyTorch、transformers、Flask

**模块**：
- `train.py` - 模型训练
- `predict.py` - 批量预测
- `server.py` - Flask API 服务端
- `client.py` - API 客户端

**启动服务**：
```bash
python server.py
```

**API 调用**：
```bash
curl -X POST http://localhost:4567/bert/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "今日头条是美国苹果公司推出的一款新闻资讯APP"}'
```

### 04_compress - BERT 模型压缩

对 BERT 模型进行压缩优化，包含三种压缩策略：

#### 知识蒸馏 (Distillation)
将大型 BERT 模型的知识迁移到轻量级 BiLSTM 学生模型。

#### 剪枝 (Pruning)
移除 BERT 模型中的冗余参数，提升推理效率。

#### 量化 (Quantization)
将 FP32 模型参数量化为 INT8，减少内存占用。

## 环境依赖

```
torch
transformers
sklearn
xgboost
fasttext
jieba
flask
tqdm
pandas
numpy
```

## 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_epoches | 3 | 训练轮数 |
| batch_size | 512 | 批次大小 |
| pad_size | 32 | 序列最大长度 |
| learning_rate | 1e-5 | 学习率 |
| device | cuda/cpu | 计算设备 |

## 快速开始

```bash
# 1. 基线模型
cd 01_baseline
python baseline.py

# 2. FastText 模型
cd 02_fasttext
python train.py

# 3. BERT 模型训练
cd 03_bert
python train.py

# 4. BERT 模型压缩
cd 04_compress
python train.py
```
