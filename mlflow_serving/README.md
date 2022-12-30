# MLflow serving


## 1. Define the project
`MLproject` file defines the project. It contains the following sections:
```yaml
name: sklearn_logistic_example # name of the project

python_env: python_env.yaml # where to get the python environment definition file

entry_points: # how to run the project
  main:
    command: "python train.py"
```


## 2. Define the python environment
python version and dependencies are defined in `python_env.yaml` file. It contains the following sections:
```yaml
python: "3.8" # which pythoh to use
build_dependencies: # how to install dependencies
  - pip
dependencies: # what dependencies to install
  - mlflow>=1.0
  - scipy
  - scikit-learn
```

## 3. Run the project
```bash
python sklearn_logistic_regression/train.py
```

> when you get errors like `Can not fine pyenv binary`, it means pyenv can not find on your machine, either not installed or not added to the path. You can install pyenv by following the instructions `https://realpython.com/intro-to-pyenv/`.

after installing, remember add pyenv to load path.
hint: append the code to .zshrc if you use zsh shell
```bash
Load pyenv automatically by adding
the following to ~/.bashrc:

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

## 4. Send request to endpoint
```bash
curl -d '{"dataframe_split": {"columns": ["x"], "data": [[1], [-1]]}}' -H 'Content-Type: application/json' -X POST localhost:5000/invocations
```


