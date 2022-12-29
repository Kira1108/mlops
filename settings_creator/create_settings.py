import logging
logging.basicConfig(level = logging.INFO)

from sklearn.datasets import load_iris
import pandas as pd
from mlobj import SklearnModelSettings

if __name__ == "__main__":
    # get training data
    iris = load_iris()
    feature_names = iris.feature_names
    df = pd.DataFrame(iris.data, columns = feature_names)
    df['add_col'] = 'somedata'

    # create all settings file
    SklearnModelSettings(
        name = 'Mymodel',
        uri = 'gs://mybucket/mymodel.pkl',
        df = df,
        version = "v0.1.1",
    ).dump_json()