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

签名就是输入输出的格式，这个东西在对接口的时候很重要，别人知道你的模型要什么，能产生什么结果，这就非常牛逼。

**基于列的模型签名**
```yaml
signature:
    inputs: '[{"name": "sepal length (cm)", "type": "double"}, {"name": "sepal width
      (cm)", "type": "double"}, {"name": "petal length (cm)", "type": "double"}, {"name":
      "petal width (cm)", "type": "double"}]'
    outputs: '[{"type": "integer"}]'

```

**基于Tensor的模型签名**

在tensorflow中，tensor就是一个基本的数据类型，包含 `name`, `shape`, `dtype`三个重要的属性和data本身。
```yaml
signature:
    inputs: '[{"name": "images", "dtype": "uint8", "shape": [-1, 28, 28, 1]}]'
    outputs: '[{"shape": [-1, 10], "dtype": "float32"}]'
```

强制Schema检查输入和signature是否一致，检查这几个方面

1. 【Name Ordering】 input name和模型签名对比，如果缺少输入，就报错，在签名中未包含的输入就忽略了，也不报错。如果在input schema中定义了 input names, 这个输入就按照signature定义的inputs的列名重新组织排序，如果输入的schema中未包含input names，就按照位置匹配签名中的列。


2. 【Input Type】input type和模型签名对比，对于列签名，进行safe type conversion，保证转换是无损的。 对于tensor签名，进行严格类型检查，不匹配就报错。


等等，等等，反正干了好多事儿。

**使用dataframe自动创建签名**
```python
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

iris = datasets.load_iris()
iris_train = pd.DataFrame(iris.data, columns=iris.feature_names)
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(iris_train, iris.target)

# 注意这个签名不是你自己写的，是用mlflow这个infer_signature函数自动生成的
# 签名传入一个输入的dataframe， 传一个输入的东西，就能自动生成签名
signature = infer_signature(iris_train, clf.predict(iris_train))

#你log模型的时候，顺手log一下签名，就牛逼了
mlflow.sklearn.log_model(clf, "iris_rf", signature=signature)
```

**手动创建签名**
如果是表格，就按照抄吧
```python
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema([
  ColSpec("double", "sepal length (cm)"),
  ColSpec("double", "sepal width (cm)"),
  ColSpec("double", "petal length (cm)"),
  ColSpec("double", "petal width (cm)"),
])
output_schema = Schema([ColSpec("long")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```
[具体log的时候，看文档吧](https://mlflow.org/docs/latest/models.html)

```bash
mlruns
└── 0
    ├── 84db702ae6dd46ffb9b3f69808a69ac9             # 第一次运行的id
    │   ├── artifacts                                # 保存的各种文件， artifacts在这儿
    │   │   └── iris_rf                              # 模型名称
    │   │       ├── conda.yaml                       # 自动生成的conda env
    │   │       ├── MLmodel                          # MLmodel也是自动生成的
    │   │       ├── model.pkl                        # 这一次运行的模型文件
    │   │       ├── python_env.yaml                  # 自动生成的python_env文件
    │   │       └── requirements.txt                 # 自动生成的requirements.txt文件
    │   ├── meta.yaml                        
    │   ├── metrics
    │   ├── params
    │   └── tags
    │       ├── mlflow.log-model.history
    │       ├── mlflow.runName
    │       ├── mlflow.source.git.commit
    │       ├── mlflow.source.name
    │       ├── mlflow.source.type
    │       └── mlflow.user
    ├── 911326f7e3c94780bf12795c154d4229
    │   ├── artifacts
    │   │   └── iris_rf
    │   │       ├── conda.yaml
    │   │       ├── MLmodel
    │   │       ├── model.pkl
    │   │       ├── python_env.yaml
    │   │       └── requirements.txt
    │   ├── meta.yaml
    │   ├── metrics
    │   ├── params
    │   └── tags
    │       ├── mlflow.log-model.history
    │       ├── mlflow.runName
    │       ├── mlflow.source.git.commit
    │       ├── mlflow.source.name
    │       ├── mlflow.source.type
    │       └── mlflow.user
    └── meta.yaml
```
















