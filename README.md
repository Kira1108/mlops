# MLOps Teseting


## 1. Concepts

[Google Mlops Article](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

[Microsoft Mlops Article](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model)

[No,you don't need mlops](https://medium.com/becoming-human/no-you-dont-need-mlops-5e1ce9fdaa4b)

> 有的时候你需要吹Mlops的牛逼，比如强行在公司找活儿， 或者强行吹嘘自己的产品。     
> 有的时候你也可以大大方方的说，过度工程化和自动化，是一件极其傻逼的事情。      
> 然而更多的时候，你需要根据自己的实际情况，选择合适的level of abstraction.    


**mlflow 搞定了文件的本地存储，远程存储，运行参数的本地存储，远程存储，以及增删改查服务。**   
**mlserver 搞定了根据元数据，模型文件，运行时等信息，把模型变成一个HTTP或者gRPC接口。**      
然后他们各自搞定了一些分布式的东西，就这样了。    

## 2. 玩一下所谓的Mlops

准备python环境
```bash
conda deactivate
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

运行代码
```bash
cd xxx-serving
```

```python
# train a model
python train***.py

# start server
mlserver start .

# run test
python client.py
```

测试json： 每个文件夹内的 `***inference_request.json` 文件



