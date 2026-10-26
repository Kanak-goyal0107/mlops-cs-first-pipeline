
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Create FastAPI application
app = FastAPI(title="Iris Model API")

# Load the trained machine learning model
model_path = Path("models/iris_model.joblib")
model = joblib.load(model_path)

# Define the input features for Iris prediction
class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Predict Iris flower species
@app.post("/predict")
def predict_iris(request: IrisRequest):
    features = [[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    class_names = ["setosa", "versicolor", "virginica"]

    return {
        "prediction": prediction,
        "class_name": class_names[prediction]
    }
