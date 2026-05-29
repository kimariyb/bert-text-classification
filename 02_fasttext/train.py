import fasttext

# Define PATH
TRAIN_DATA_PATH = "./data/train_processed.csv"
VAL_DATA_PATH = "./data/val_processed.csv"
TEST_DATA_PATH = "./data/test_processed.csv"
MODEL_SAVE_PATH = "./model/toutiao_fasttext.bin"

model = fasttext.train_supervised(
    input=TRAIN_DATA_PATH,
    autotuneValidationFile=VAL_DATA_PATH,
    autotuneDuration=6,
    wordNgrams=2, verbose=3)

print("The number of words:", len(model.words))
print("Label values:", model.labels)

res = model.test(TEST_DATA_PATH)
print(res)

# Save model
model.save_model(MODEL_SAVE_PATH)
# (10000, 0.9082, 0.9082)
