# MLOps Teseting

Prepare environment
```bash
conda deactivate
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Run Code
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