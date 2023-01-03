# MLflow models

## 1. Folder Structure

MLflow model是一个文件夹，里面有很多的文件，包含一个MLmodel文件在模型文件夹的根目录。
这个Mlmodel文件定义了模型的`flavor`,有了`flavor`以后，部署工具就知道怎么部署这个模型了。       


```bash
my_model/
├── MLmodel                  # 模型定义，元数据等 [需要琢磨一下]
├── model.pkl                # 真正的模型文件 [模型训练结果]
├── conda.yaml               # conda 环境 - 自动创建
├── python_env.yaml          # virtualenv环境 - 自动创建
└── requirements.txt         # pip依赖 - 自动创建
```
---
## 2. Model Description
`MLmodel`
```yaml
time_created: 2018-05-25T17:28:53.35  # 模型的创建时间

flavors:                              # 这个就是flavor定义
  sklearn:                            # 告诉别人，这个是sklearn的模型
    sklearn_version: 0.19.1           # sklearn用的哪个版本
    pickled_model: model.pkl          # 模型文件的未知
  python_function:                    # 这个东西大概是用一个纯python也能运行的东西
    loader_module: mlflow.sklearn     # 从mlflow用这个包搞
```

其实可以有更多的东西的
- `time_created`: 创建时间
- `run_id`: 运行id
- `signagure`: 模型签名
- `input_example`: 输入示例
- `databrices_runtime`: 如果是再databricks平台训练的，可以用这个
- `model_version`: 模型版本

*Note*:

> 里面`signature`是比较重要的，因为它定义了模型的的输入和输出参数， 这些东西再请求参数和python高层对象转换的时候，是必须的。    

> conda, pyenv啥的，是模型运行时需要的环境，与MLproject中定义的环境不是一个东西，当你用mlflow去log你的model的时候，这些环境文件就自动生成了。
---
## 3. Environments
Log model的时候，把condaenv搞一下
> log model的时候，你可以传一个conda env， 你也可以不传，然后mlflow就自动给你创建一个。   

```python
conda_env = {
    'channels': ['conda-forge'],
    'dependencies': [
        'python=3.8.8',
        'pip'],
    'pip': [
        'mlflow',
        'scikit-learn==0.23.2',
        'cloudpickle==1.6.0'
    ],
    'name': 'mlflow-env'
}
mlflow.sklearn.log_model(model, "my_model", conda_env=conda_env)
```

然后就创建了`conda.yaml`, `python_env.yaml`和`requirements.txt`三个文件


```bash
mlflow models serve -m my_model

mlflow deployments create -t sagemaker -m my_model [other options]
```

---
## 4. Signature












