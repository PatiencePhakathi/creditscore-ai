import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def train_models(X_train, y_train, preprocessor):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    trained_pipelines = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

    return trained_pipelines

def save_model(pipeline, model_name="credit_model.joblib"):
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", model_name)
    joblib.dump(pipeline, path)
    return path
