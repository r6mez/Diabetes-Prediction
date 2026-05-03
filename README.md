# Diabetes Prediction Web Application

This is a simple web application built using FastAPI that allows users to input their health data and receive a prediction on whether they are likely to have diabetes based on a machine learning model.

Data: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## Running the Web App

1. Install dependencies:

   ```bash
   pip install fastapi uvicorn pandas scikit-learn joblib pydantic
   ```

2. Start the server from the [web/](web/) directory (model paths are relative):

   ```bash
   cd web
   fastapi dev main.py
   ```

3. Open <http://127.0.0.1:8000> in your browser to use the form, or POST JSON to `/predict` to get predictions from all 8 models.

## Notebook Workflow

The notebook at [notebook/practical-project.ipynb](notebook/practical-project.ipynb) covers:

### Preprocessing

- Loaded the Pima Indians Diabetes dataset (768 rows, 9 columns).
- Replaced invalid `0` values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` with `NaN`, then imputed them with the median per `Outcome` group.
- Capped outliers using the IQR method (1.5 × IQR clipping) on all feature columns.
- Rounded integer-valued columns back to `int`.
- Split into 80% train / 20% test with stratification on `Outcome` (614 train, 154 test).

### Models Trained

| Model                  | Test Accuracy |
| ---------------------- | ------------- |
| AdaBoost               | 0.864         |
| Random Forest          | 0.857         |
| SVM (RBF kernel)       | 0.851         |
| KNN (best K=7, scaled) | 0.844         |
| Decision Tree          | 0.818         |
| Logistic Regression    | 0.753         |
| LDA                    | 0.753         |
| Naive Bayes            | 0.747         |

KNN used `StandardScaler` and 5-fold cross-validation to select the best K (1–30).

### Persistence

All trained models plus the `StandardScaler` are saved as `.pkl` files in [models/](models/) for use by the FastAPI web app.
