import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    # plain old training code
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    
    # log metric and model to mlflow
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
    
    # print some info about the run
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)