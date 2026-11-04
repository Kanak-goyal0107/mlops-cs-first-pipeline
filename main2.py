from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import os
from contextlib import asynccontextmanager
from typing import Callable


#app=FastAPI(title="Iris Model API")

# MLFlow Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI","http://127.0.0.1:6001"
)

MODEL_URI = "models:/iris-classifier@champion"




#model_path = Path("models/iris_model.joblib")
#model = joblib.load(model_path)
def load_mlflow_model():
    #Tell the MLFLOW client where the model registry lines
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    retutn mlflow.sklearn.load_model(MODEL_URI)


class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

#Application Factory
def create_app(model_loader: Callable=load_mlflow_model, version_loader: Callable=None):

    if version_loader is None:
        def version_loader():
            client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            champion = client.get_model_version_by_alias(
                name="iris_classifier"
                alias="champion"
            )
            return champion.version
    # Lifespan function executes when the application starts
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        #Load the model once during startup
        # Injected loader can be real MLFlow or a test model
        try:
            app.state.model = model_loader()
        except Exception as ex:
            raise RuntimeError(
                f"Failed to load MLFlow champion model..\n{ex}"
            )
        client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        champion = client.get_model_version_by_alias(
            name="iris-classifier",
            alias="champion"
        )
        app.state.model_version = champion.version
        print(f"Loaded iris-classifier version {app.state.model_version}")
        yield
    return app


@app.get("/")
def health_check():
    return {"message": "API is ready to serve..."}

@app.get("/health")
def health_check():
    return {"status": "healthy",
    "model_name": "iris_classifier",
    "model_version": app.state.model_version
    }


@app.post("/predict")
def predict(data: IrisRequest, request: Request):
    model=request.app.stae.model
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "prediction": prediction,
        "class_name": ["setosa", "versicolor", "virginica"][prediction]
    }