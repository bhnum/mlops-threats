from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import mlflow
# from mlflow.tracking import MlflowClient
from prefect import flow

mlflow.set_experiment("Test")


# def yield_artifacts(run_id, path=None):
#     """Yield all artifacts in the specified run"""
#     client = MlflowClient()
#     for item in client.list_artifacts(run_id, path):
#         if item.is_dir:
#             yield from yield_artifacts(run_id, item.path)
#         else:
#             yield item.path


# def fetch_logged_data(run_id):
#     """Fetch params, metrics, tags, and artifacts in the specified run"""
#     client = MlflowClient()
#     data = client.get_run(run_id).data
#     # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
#     tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
#     artifacts = list(yield_artifacts(run_id))
#     return {
#         "params": data.params,
#         "metrics": data.metrics,
#         "tags": tags,
#         "artifacts": artifacts,
#     }


@flow(log_prints=True)
def train():
    print('Model training started')

    mlflow.sklearn.autolog()

    iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

    features = ["sepal_length","sepal_width", "petal_length", "petal_width"]
    target = ["species"]

    X_train, X_test, y_train, y_test = train_test_split(iris[features], iris[target].values, test_size=0.33)

    forest = RandomForestClassifier()

    with mlflow.start_run() as run:
        forest.fit(X_train, y_train)



        run_id = run.info.run_id
        print(f"Logged data and model in run: {run_id}")

        # # show logged data
        # for key, data in fetch_logged_data(run_id).items():
        #     print(f"\n---------- logged {key} ----------")
        #     pprint(data)
    
    print('Model training finished')


if __name__ == "__main__":
    train()
