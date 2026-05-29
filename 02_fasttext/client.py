import requests
import time

url = "http://0.0.0.0:5000/v1/main"
data = {
    "uid": "20260413-1",
    "text": "这是一条测试消息。"
}

start = time.time()

res = requests.post(url, data=data)
print("Input: ", data['text'])
print("Result: ", res.text)
print("Time: ", time.time() - start)

# Input:  这是一条测试消息。
# Result:  __label__stocks
# Time:  0.011455774307250977
