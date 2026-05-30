from predict import inference
from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/bert/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    result = inference(data)    
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4567, debug=True)
