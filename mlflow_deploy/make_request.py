import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

json_format = df.head(5).to_json(orient='split', index=False)

with open("sample_request.json", "w") as f:
    f.write(json_format)
    
    
    
    
print("Generate sample request success.")



