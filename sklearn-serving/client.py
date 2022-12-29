from sklearn import datasets
from sklearn.model_selection import train_test_split
import requests

digits = datasets.load_digits()


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

x_0 = X_test[0:1]
inference_request = {
    "inputs": [
        {
          "name": "predict",
          "shape": x_0.shape,
          "datatype": "FP32",
          "data": x_0.tolist()
        }
    ]
}

endpoint = "http://localhost:8080/v2/models/mnist-svm/versions/v0.1.0/infer"

import json

json.dump(inference_request, open("inference_request.json", "w"))

response = requests.post(endpoint, json=inference_request)

print(response.json())