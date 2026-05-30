from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from scipy.stats import randint, uniform

import pandas as pd
import joblib
import os

# ===================== 路径配置 =====================
TRAIN_DATA_PATH   = "./data/train_processed.csv"
TEST_DATA_PATH    = "./data/test_processed.csv"
STOP_WORDS_PATH   = "../dataset/stopwords.txt"
MODEL_OUTPUT_DIR  = "./models"

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ===================== 加载数据 =====================
train_df = pd.read_csv(TRAIN_DATA_PATH, sep="\t")
test_df  = pd.read_csv(TEST_DATA_PATH,  sep="\t")

with open(STOP_WORDS_PATH, "r", encoding="utf-8") as f:
    stop_words = [line.strip() for line in f]

# ===================== TF-IDF 向量化 =====================
vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    ngram_range=(1, 2),
    max_features=10000,
)
X_train = vectorizer.fit_transform(train_df["words"].values)
y_train = train_df["label"]

X_test = vectorizer.transform(test_df["words"].values)
y_test  = test_df["label"]

# ===================== 并发核心数配置 =====================
# CPU 核心数，RandomizedSearchCV 的 n_jobs 用这个
# 根据你的机器调整，一般设为物理核心数即可
N_CV_JOBS = 32

# ===================== 1. 随机森林 =====================
# 中等数据量下深度不宜过大，否则容易过拟合且极慢
param_dist_rf = {
    "n_estimators":      randint(100, 300),
    "max_depth":         randint(10, 30),    
    "min_samples_split": randint(2, 10),
    "min_samples_leaf":  randint(1, 5),
    "max_features":      ["sqrt", "log2"],
}

random_search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=N_CV_JOBS),
    param_dist_rf,
    n_iter=20,
    cv=5,
    scoring="f1_weighted",
    n_jobs=1,
    random_state=42,
    verbose=2,
)

# ===================== 2. XGBoost（GPU 加速） =====================
param_dist_xgb = {
    "n_estimators":      randint(100, 300),
    "max_depth":         randint(3, 8),          # 原 10~40 → 大幅收窄
    "min_child_weight":  randint(1, 6),
    "gamma":             uniform(0, 5),           # 原 randint(0,10) → 连续采样更合理
    "subsample":         uniform(0.5, 0.5),       # 0.5~1.0
    "colsample_bytree":  uniform(0.5, 0.5),       # 0.5~1.0
    "learning_rate":     uniform(0.01, 0.29),     # 0.01~0.30，原上限 0.1 太窄
    "reg_alpha":         uniform(0, 1),           # 新增：L1 正则
    "reg_lambda":        uniform(1, 4),           # 新增：L2 正则
}

random_search_xgb = RandomizedSearchCV(
    XGBClassifier(
        random_state=42,
        verbosity=0,
        device="cuda",      # GPU 加速
        nthread=1,          # ← 关键：关闭 XGBoost CPU 多线程，交给 sklearn 并发
        eval_metric="mlogloss",
    ),
    param_dist_xgb,
    n_iter=20,
    cv=5,
    scoring="f1_weighted",
    n_jobs=1,               # ← GPU 模式下 sklearn 并发设为 1，避免多进程争抢显存
    random_state=42,
    verbose=2,
)

# ===================== 训练 & 评估 =====================
def evaluate_model(model, model_name: str):
    print(f"\n{'='*50}")
    print(f"开始训练：{model_name}")
    print(f"{'='*50}")
    model.fit(X_train, y_train)

    print(f"\n{model_name} 最优参数：")
    for k, v in model.best_params_.items():
        print(f"  {k}: {v}")

    y_pred = model.best_estimator_.predict(X_test)

    acc       = accuracy_score(y_test,  y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test,    y_pred, average="weighted",  zero_division=0)
    f1        = f1_score(y_test,        y_pred, average="weighted",  zero_division=0)

    print(f"\n{model_name} 测试集结果：")
    print(f"  准确率：{acc:.4f}")
    print(f"  精确率：{precision:.4f}")
    print(f"  召回率：{recall:.4f}")
    print(f"  F1 分数：{f1:.4f}")

    return model.best_estimator_, f1


rf_model,  rf_f1  = evaluate_model(random_search_rf,  "随机森林")
xgb_model, xgb_f1 = evaluate_model(random_search_xgb, "XGBoost")

# ===================== 选择最优模型 =====================
print(f"\n{'='*50}")
print(f"随机森林 F1：{rf_f1:.4f}")
print(f"XGBoost   F1：{xgb_f1:.4f}")

best_model      = rf_model  if rf_f1 > xgb_f1 else xgb_model
best_model_name = "随机森林" if rf_f1 > xgb_f1 else "XGBoost"
print(f"最优模型：{best_model_name}")

# ===================== 保存 =====================
joblib.dump(best_model,  f"{MODEL_OUTPUT_DIR}/best_model.pkl")
joblib.dump(vectorizer,  f"{MODEL_OUTPUT_DIR}/tfidf_vectorizer.pkl")
print(f"\n模型已保存至 {MODEL_OUTPUT_DIR}/")