from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from scipy.stats import randint

import pandas as pd
import joblib

# Define PATH
TRAIN_DATA_PATH = "./data/train_processed.csv"
TEST_DATA_PATH = "./data/test_processed.csv"
STOP_WORDS_PATH = "./data/stopwords.txt"

# Load data
train_df = pd.read_csv(TRAIN_DATA_PATH, sep="\t")
test_df = pd.read_csv(TEST_DATA_PATH, sep="\t")

# 构建预料库
corpus = train_df['words'].values

# Load stop words
with open(STOP_WORDS_PATH, "r", encoding="utf-8") as f:
    stop_words = [line.strip() for line in f.readlines()]

# Vectorize
vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2), max_features=10000)
X_train = vectorizer.fit_transform(corpus)
y_train = train_df['label']

X_test = vectorizer.transform(test_df['words'].values)
y_test = test_df['label']

param_dist = {
    "n_estimators": randint(100, 400),
    "max_depth": randint(10, 40),
    "min_samples_split": randint(2, 6),
    "min_samples_leaf": randint(1, 4),
    "max_features": ["sqrt", "log2"]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=32),
    param_dist,
    n_iter=20,
    cv=5,
    scoring="f1_weighted",
    n_jobs=32,
    random_state=42,
    verbose=2
)

print("开始搜索最优参数...")
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

print("\n===== 最优参数 =====")
print(random_search.best_params_)

y_pred = best_model.predict(X_test)

# ===================== 计算评价指标 =====================
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n===== 测试集评估结果 =====")
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"F1分数   (F1):      {f1:.4f}")

joblib.dump(best_model, "./models/best_rf_model.pkl")
joblib.dump(vectorizer, "./models/tfidf_vectorizer.pkl")
print("\n模型与TF-IDF向量器已保存！")

"""
===== 最优参数 =====
{'max_depth': 34, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 336}

===== 测试集评估结果 =====
准确率 (Accuracy):  0.7678
精确率 (Precision): 0.7935
召回率 (Recall):    0.7678
F1分数   (F1):      0.7745
"""