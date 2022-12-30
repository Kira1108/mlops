# MLflow Simple Tutorial

## Write model fitting code.
It is as simple as insert model fitting code into a tracking template.
```python
with mlflow.start_run():
    # fit the model
    ...

    # evaluate model
    ...

    # log hypyter parameters
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)

    # log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    # log model (if model tracking with file, register model is not supported)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
    else:
        mlflow.sklearn.log_model(lr, "model")

```

## Write a MLproject file
```yaml

name: tutorial

conda_env: conda.yaml # points to conda environment definition file

entry_points:
  main:
    parameters:  # this model taks 2 parameters, alpha and l1_ratio
      alpha: {type: float, default: 0.5}
      l1_ratio: {type: float, default: 0.1}
    command: "python train.py {alpha} {l1_ratio}"  # command used to run the model
```

## Write Environment definition file `conda.yaml`

```yaml
name: tutorial
channels:
  - conda-forge
dependencies:
  - python=3.8
  - pip
  - pip:
      - scikit-learn==0.23.2
      - mlflow>=1.0
      - pandas
```

## Run the project

mlflow start a new environment deifned in conda.yaml and run the project with `command` defined in MLproject file.
```bash
# run local project
mlflow run sklearn_elasticnet_wine -P alpha=0.42

# run remote project
mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5.0
```