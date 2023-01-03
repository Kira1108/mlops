# MLflow Deployment

**endpoints**

- `/ping`          ：健康检查
- `/health`        ：和`/ping`一样
- `/invocations`   ：模型预测
- `/version`       ：获取mlflow的版本

REST API接收csv或者json输入， 具体哪种类型要在 `Content-Type` 中指定。header 的值为 `application/json` 或者 `application/csv`。    

**Csv file**     
csv必须是一个有效的pd.DataFrame.


**Json**    
json必须是下面几种测试中的一种


- `dataframe_split`     
```bash
curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_split": {
      "columns": ["a", "b", "c"],
      "data": [[1, 2, 3], [4, 5, 6]]
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
