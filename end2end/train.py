from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import os
import json
import pandas as pd
from mlserver.codecs import PandasCodec
import joblib


iris = load_iris()
y = iris['target']
X = iris['data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# classifier = svm.SVC(gamma=0.001)

# classifier.fit(X_train, y_train)

# model_file_name = "iris_sk.joblib"
# joblib.dump(classifier, model_file_name)


feature_names = iris.feature_names
df = pd.DataFrame(X, columns = feature_names)

testdata = pd.DataFrame(X_test, columns = feature_names)
test_request = PandasCodec.encode_request(testdata.head(1))
print(test_request)