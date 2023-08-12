from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from prometheus_fastapi_instrumentator import Instrumentator
from mlflow import MlflowClient
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

instrumentator = Instrumentator().instrument(app)


def fetch_latest_model():
    client = MlflowClient()
    return client.get_latest_versions()[0].name


def fetch_latest_version(model_name):
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/Production"
    )
    return model


@app.on_event("startup")
async def startup():
    instrumentator.expose(app)


@app.get("/predict/")
def model_output(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    print("Works I")
    model_name = 'Unnamed'
    model = fetch_latest_version(model_name)
    print("Works II")
    input = pd.DataFrame({"sepal_length": [sepal_length], "sepal_width": [sepal_width], "petal_length": [petal_length], "petal_width": [petal_width]})

    prediction = model.predict(input)
    print(prediction)
    return {"prediction": prediction[0]}
