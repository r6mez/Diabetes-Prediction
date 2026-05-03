import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from joblib import load
from pathlib import Path
from pydantic import BaseModel, Field

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

lr_model = load("../models/logistic_regression.pkl")
lda_model = load("../models/linear_discriminant_analysis.pkl")
ada_model = load("../models/adaboost.pkl")
rf_model = load("../models/random_forest.pkl")
dt_model = load("../models/decision_tree.pkl")
knn_model = load("../models/k_nearest_neighbors.pkl")
scaler = load("../models/scaler.pkl")
nb_model = load("../models/naive_bayes.pkl")
svm_model = load("../models/support_vector_machine.pkl")

class PersonData(BaseModel):
    pregnancies: int = Field(gt=0, lt=25)
    glucose: int = Field(gt=0, lt=600)
    blood_pressure: int = Field(gt=0, lt=200)
    skin_thickness: int = Field(gt=0, lt=100)
    insulin: int = Field(gt=0, lt=1000)
    bmi: float = Field(gt=0, lt=100)
    diabetes_pedigree_function: float = Field(gt=0, lt=10)
    age: int = Field(gt=0, lt=150)

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict")
async def predict(person_data: PersonData):
    X = pd.DataFrame([list(person_data.model_dump().values())], columns=FEATURE_NAMES)
    lr_prediction = lr_model.predict(X)
    lda_prediction = lda_model.predict(X)
    ada_prediction = ada_model.predict(X)
    rf_prediction = rf_model.predict(X)
    dt_prediction = dt_model.predict(X)
    knn_prediction = knn_model.predict(scaler.transform(X))
    nb_prediction = nb_model.predict(X)
    svm_prediction = svm_model.predict(X)
    
    return {
        "predictions": [
            {"model": "Logistic Regression", "prediction": int(lr_prediction[0])},
            {"model": "Linear Discriminant Analysis", "prediction": int(lda_prediction[0])},
            {"model": "AdaBoost", "prediction": int(ada_prediction[0])},
            {"model": "Random Forest", "prediction": int(rf_prediction[0])},
            {"model": "Decision Tree", "prediction": int(dt_prediction[0])},
            {"model": "K-Nearest Neighbors", "prediction": int(knn_prediction[0])},
            {"model": "Naive Bayes", "prediction": int(nb_prediction[0])},
            {"model": "Support Vector Machine", "prediction": int(svm_prediction[0])}
        ]
    }