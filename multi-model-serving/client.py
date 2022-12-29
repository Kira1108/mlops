import requests
import os
import xgboost as xgb
import requests

from urllib.parse import urlparse
from sklearn.datasets import load_svmlight_file
from sklearn import datasets
from sklearn.model_selection import train_test_split


def test_sklearn():
    digits = datasets.load_digits()

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    X_train, X_test_digits, y_train, y_test_digits = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False)
    
    x_0 = X_test_digits[0:1]
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
    response = requests.post(endpoint, json=inference_request)

    print(response.json())


TRAIN_DATASET_URL = 'https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.train'
TEST_DATASET_URL = 'https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.test'


def _download_file(url: str) -> str:
    parsed = urlparse(url)
    file_name = os.path.basename(parsed.path)
    file_path = os.path.join(os.getcwd(), file_name)
    
    if os.path.exists(file_path):
        return file_path
    
    res = requests.get(url)
    
    with open(file_path, 'wb') as file:
        file.write(res.content)
    
    return file_path

def test_xgboost():
    train_dataset_path = _download_file(TRAIN_DATASET_URL)
    test_dataset_path = _download_file(TEST_DATASET_URL)

    X_train, y_train = load_svmlight_file(train_dataset_path)
    X_test_agar, y_test_agar = load_svmlight_file(test_dataset_path)
    X_train = X_train.toarray()
    X_test_agar = X_test_agar.toarray()
    
    x_0 = X_test_agar[0:1]
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

    endpoint = "http://localhost:8080/v2/models/mushroom-xgboost/versions/v0.1.0/infer"
    response = requests.post(endpoint, json=inference_request)

    print(response.json())
        
if __name__== "__main__":
    print("Testing sklearn model: ", end = "")
    test_sklearn()
    print("Done")
    
    print("Testing xgboost model: ", end = "")
    test_xgboost()
    print("Done")
    
    