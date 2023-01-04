# MLflow Deployment

## 1. Endpoints

- `/ping`          ：健康检查
- `/health`        ：和`/ping`一样
- `/invocations`   ：模型预测
- `/version`       ：获取mlflow的版本

REST API接收csv或者json输入， 具体哪种类型要在 `Content-Type` 中指定。header 的值为 `application/json` 或者 `application/csv`。    


## 2. Request Parameters
**Csv file**     
csv必须是一个有效的pd.DataFrame.


**Json**    
json必须是下面几种测试中的一种


- `dataframe_split`     
```bash
curl http://127.0.0.1:8083/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_split": {
    "columns": ["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"],
    "data": [
        [5.1,3.5,1.4,0.2],
        [4.9,3.0,1.4,0.2],
        [4.7,3.2,1.3,0.2],
        [4.6,3.1,1.5,0.2],
        [5.0,3.6,1.4,0.2]
    ]
  }
}'
```



- `dataframe_records`     
```bash
curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_records": [
    {"a": 1,"b": 2,"c": 3},
    {"a": 4,"b": 5,"c": 6}
  ]
}'
```

- `instances` 
```bash
# numpy/tensor input using TF serving's "instances" format
curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
    "instances": [
        {"a": "s1", "b": 1, "c": [1, 2, 3]},
        {"a": "s2", "b": 2, "c": [4, 5, 6]},
        {"a": "s3", "b": 3, "c": [7, 8, 9]}
    ]
}'
```

- `inputs` 
```bash
# numpy/tensor input using TF serving's "inputs" format
curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
    "inputs": {"a": ["s1", "s2", "s3"], "b": [1, 2, 3], "c": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
}'
```

## 3. Model Serving Document

本地部署
```bash
mlflow models serve -m runs:/98a724ca0f0f46c0b927cff184427630/iris_rf -h 0.0.0.0 -p 8081 -w 1 

nohup mlflow models serve -m runs:/98a724ca0f0f46c0b927cff184427630/iris_rf -h 0.0.0.0 -p 8083 -w 1 &
```

命令行参数
```bash
> mlflow models serve --help  
                                                                                                                                                    ─
Usage: mlflow models serve [OPTIONS]

  Serve a model saved with MLflow by launching a webserver on the specified
  host and port. The command supports models with the ``python_function`` or
  ``crate`` (R Function) flavor. For information about the input data formats
  accepted by the webserver, see the following documentation:
  https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools.

  .. warning::

      Models built using MLflow 1.x will require adjustments to the endpoint
      request payload     if executed in an environment that has MLflow 2.x
      installed. In 1.x, a request payload     was in the format:
      ``{'columns': [str], 'data': [[...]]}``. 2.x models require     payloads
      that are defined by the structural-defining keys of either
      ``dataframe_split``,     ``instances``, ``inputs`` or
      ``dataframe_records``. See the examples below for     demonstrations of
      the changes to the invocation API endpoint in 2.0.

  .. note::

      Requests made in pandas DataFrame structures can be made in either
      `split` or `records`     oriented formats.     See https://pandas.pydata
      .org/docs/reference/api/pandas.DataFrame.to_json.html for     detailed
      information on orientation formats for converting a pandas DataFrame to
      json.
  

Options:
  -m, --model-uri URI    URI to the model. A local path, a 'runs:/' URI, or a
                         remote storage URI (e.g., an 's3://' URI). For more
                         information about supported remote URIs for model
                         artifacts, see https://mlflow.org/docs/latest/trackin
                         g.html#artifact-stores  [required]
  -p, --port INTEGER     The port to listen on (default: 5000).
  -h, --host HOST        The network address to listen on (default:
                         127.0.0.1). Use 0.0.0.0 to bind to all addresses if
                         you want to access the tracking server from other
                         machines.
  -t, --timeout INTEGER  Timeout in seconds to serve a request (default: 60).
  -w, --workers TEXT     Number of gunicorn worker processes to handle
                         requests (default: 4).
  --env-manager TEXT     If specified, create an environment for MLmodel using
                         the specified environment manager. The following
                         values are supported:
                         
                         - local: use the local environment
                         - virtualenv: use virtualenv (and pyenv for Python version management)
                         - conda: use conda
                         
                         If unspecified, default to virtualenv.
  --no-conda             If specified, use local environment.
  --install-mlflow       If specified and there is a conda or virtualenv
                         environment to be activated mlflow will be installed
                         into the environment after it has been activated. The
                         version of installed mlflow will be the same as the
                         one used to invoke this command.
  --enable-mlserver      Enable serving with MLServer through the v2 inference
                         protocol.
  --help                 Show this message and exit.
```


