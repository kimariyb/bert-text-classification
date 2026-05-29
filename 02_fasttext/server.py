import fasttext
import jieba
from flask import Flask, request

app = Flask(__name__)

jieba.load_userdict("./data/stopwords.txt")

try:
    model = fasttext.load_model("./model/toutiao_fasttext.bin")
    print("Load model done.")
except Exception as e:
    print(e)


@app.route('/v1/main', methods=['POST'])
def main():
    uid = request.form.get('uid', '')
    text = request.form.get('text', '')

    input_text = ' '.join(jieba.lcut(text))

    res = model.predict(input_text)
    predict_name = res[0][0]
    print(predict_name)

    return predict_name


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
