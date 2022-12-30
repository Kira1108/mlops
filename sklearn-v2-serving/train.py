from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
import joblib
from settings_creator import SklearnModelSettings, create_request

iris = load_iris()
y = iris['target']
X = iris['data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

classifier = svm.SVC(gamma=0.001)

classifier.fit(X_train, y_train)

model_file_name = "iris_sk.joblib"
joblib.dump(classifier, model_file_name)

feature_names = iris.feature_names
df = pd.DataFrame(X, columns = feature_names)


SklearnModelSettings(
        name = 'iris_sk',
        uri = 'iris_sk.joblib',
        df = df,
        version = "v0.1.0",
    ).dump_json()


create_request(df, n = 3)





