# MLflow tracking API


## Log whatever you want to track with MLflow
```python
from mlflow import log_metric, log_param, log_artifact

# log hypter parameters
log_param("param1", 5)

# log evaluations
log_metric("foo", 1)

# log folder of files
log.artifact("output")
```

## Log to remote tracking server
Set mlflow tracking uri and experiment name
```python
import mlflow
mlflow.set_tracking_uri("http://YOUR-SERVER:4040")
mlflow.set_experiment("my-experiment")

```

## Run MLflow project
It takes long time to install environment and run the project.
```bash
# run local project
mlflow run sklearn_elasticnet_wine -P alpha=0.5

# run remote project
mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5.0
```
