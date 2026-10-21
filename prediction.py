import pandas as pd

def make_prediction(pipeline, input_dict, feature_columns):
    input_df = pd.DataFrame([input_dict])[feature_columns]
    pred_class = pipeline.predict(input_df)[0]
    pred_proba = pipeline.predict_proba(input_df)[0][1] if hasattr(pipeline, "predict_proba") else None

    if pred_proba is not None:
        if pred_proba < 0.30:
            risk_category = "Low Risk (Tier A)"
        elif pred_proba < 0.60:
            risk_category = "Moderate Risk (Tier B)"
        else:
            risk_category = "High Risk (Tier C)"
    else:
        risk_category = "Default Risk Positive" if pred_class == 1 else "Default Risk Negative"

    return int(pred_class), float(pred_proba) if pred_proba is not None else 0.0, risk_category
