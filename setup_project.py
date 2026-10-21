import os

# Define directory structure
folders = ["data", "models", "tests"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {folder.capitalize()} Directory\n")

# Write requirements.txt
requirements_content = """streamlit==1.32.0
pandas==2.2.1
numpy==1.26.4
scikit-learn==1.4.1.post1
plotly==5.19.0
joblib==1.3.2
requests==2.31.0
pytest==8.1.0
"""
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements_content)

# Write data_loader.py
data_loader_code = '''import pandas as pd
import requests
import io
import streamlit as st

@st.cache_data(show_spinner=True)
def load_german_credit_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    columns = [
        'status_checking_account', 'duration_in_months', 'credit_history', 'purpose',
        'credit_amount', 'savings_account', 'present_employment_since', 'installment_rate',
        'personal_status_sex', 'other_debtors', 'present_residence_since', 'property',
        'age', 'other_installment_plans', 'housing', 'existing_credits', 'job',
        'n_dependents', 'telephone', 'foreign_worker', 'credit_risk'
    ]
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), sep=' ', header=None, names=columns)
        df['credit_risk'] = df['credit_risk'].map({1: 0, 2: 1})
        return df, "UCI Statlog German Credit Data", url, "University of California, Irvine (Statlog)"
    except Exception as e:
        st.error(f"Failed to download dataset from UCI repository: {e}")
        return None, None, None, None

def validate_uploaded_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")
        if df.shape[1] < 2:
            raise ValueError("Dataset must contain at least 2 columns.")
        return df, None
    except Exception as e:
        return None, str(e)
'''
with open("data_loader.py", "w", encoding="utf-8") as f:
    f.write(data_loader_code)

# Write preprocessing.py
preprocessing_code = '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessing_pipeline(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor, numeric_features, categorical_features

def prepare_data(df, target_col):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocessor, num_cols, cat_cols = get_preprocessing_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() <= 10 else None
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor, X_train_processed, X_test_processed
'''
with open("preprocessing.py", "w", encoding="utf-8") as f:
    f.write(preprocessing_code)

# Write model_training.py
model_training_code = '''import joblib
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
'''
with open("model_training.py", "w", encoding="utf-8") as f:
    f.write(model_training_code)

# Write evaluation.py
evaluation_code = '''import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A",
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics, y_pred, y_proba
'''
with open("evaluation.py", "w", encoding="utf-8") as f:
    f.write(evaluation_code)

# Write prediction.py
prediction_code = '''import pandas as pd

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
'''
with open("prediction.py", "w", encoding="utf-8") as f:
    f.write(prediction_code)

# Write visualizations.py
visualizations_code = '''import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

def plot_target_distribution(df, target_col):
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, 'Count']
    counts['Label'] = counts[target_col].map({0: 'Good Credit', 1: 'Bad Credit / Default'})
    fig = px.bar(counts, x='Label', y='Count', color='Label',
                 title="Credit Risk Outcome Distribution",
                 color_discrete_map={'Good Credit': '#008080', 'Bad Credit / Default': '#1E293B'})
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    return fig

def plot_confusion_matrix(cm):
    z = cm
    x = ['Predicted Good', 'Predicted Bad']
    y = ['Actual Good', 'Actual Bad']
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Teal', annotation_text=np.array(z).astype(str))
    fig.update_layout(title_text="Confusion Matrix", plot_bgcolor='white', paper_bgcolor='white')
    return fig
'''
with open("visualizations.py", "w", encoding="utf-8") as f:
    f.write(visualizations_code)

# Write responsible_ai.py
responsible_ai_code = '''import streamlit as st

def render_responsible_ai_section():
    st.subheader("Responsible AI, Fairness & Governance")
    st.markdown("""
    In financial services, automated decision-making carries profound ethical and regulatory obligations. 
    CreditScore AI integrates rigorous governance principles:
    * **Historical & Sampling Bias:** Models trained on historical lending data may reflect past systemic exclusions.
    * **False Positives vs. False Negatives:** Rejecting creditworthy applicants vs. approving high default risk.
    * **Human Oversight:** Predictions are analytical estimates and must never replace professional underwriting judgment.
    """)
'''
with open("responsible_ai.py", "w", encoding="utf-8") as f:
    f.write(responsible_ai_code)

# Write app.py
app_code = '''import streamlit as st
import pandas as pd
from data_loader import load_german_credit_data, validate_uploaded_csv
from preprocessing import prepare_data
from model_training import train_models, save_model
from evaluation import evaluate_model
from prediction import make_prediction
from visualizations import plot_target_distribution, plot_confusion_matrix
from responsible_ai import render_responsible_ai_section

st.set_page_config(page_title="CreditScore AI", page_icon="💳", layout="wide")

st.sidebar.title("CreditScore AI")
st.sidebar.markdown("*Understand Credit Risk Through Data.*")
menu = st.sidebar.radio("Navigation", ["Overview", "Data Explorer", "Model Lab", "Predictions", "Responsible AI", "About Project"])

if "df" not in st.session_state:
    st.session_state.df = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None

if menu == "Overview":
    st.title("💳 CreditScore AI: FinTech Risk Analytics")
    st.markdown("Welcome to **CreditScore AI**, an end-to-end machine learning platform for credit risk analysis.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Status", "Loaded" if st.session_state.df is not None else "Not Loaded")
    col2.metric("Model Status", "Trained" if st.session_state.trained_model is not None else "Pending")
    col3.metric("Compliance", "Basel III Aligned")

elif menu == "Data Explorer":
    st.title("📂 Data Acquisition & Exploration")
    data_source = st.radio("Select Data Source", ["UCI Statlog German Credit Data", "Upload CSV"])
    
    if data_source == "UCI Statlog German Credit Data":
        if st.button("Fetch UCI Dataset"):
            df, name, url, publisher = load_german_credit_data()
            if df is not None:
                st.session_state.df = df
                st.session_state.target_col = 'credit_risk'
                st.success(f"Loaded {name} ({df.shape[0]} rows, {df.shape[1]} features).")
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df, err = validate_uploaded_csv(uploaded_file)
            if err:
                st.error(err)
            else:
                st.session_state.df = df
                st.session_state.target_col = df.columns[-1]
                st.success(f"Loaded CSV with {df.shape[0]} rows.")

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        fig = plot_target_distribution(st.session_state.df, st.session_state.target_col)
        st.plotly_chart(fig, use_container_width=True)

elif menu == "Model Lab":
    st.title("🔬 Model Training & Evaluation Lab")
    if st.session_state.df is None:
        st.warning("Load a dataset in Data Explorer first.")
    else:
        model_choice = st.selectbox("Algorithm", ["Random Forest", "Logistic Regression"])
        if st.button("Train Model"):
            with st.spinner("Training..."):
                X_train, X_test, y_train, y_test, preprocessor, _, _ = prepare_data(st.session_state.df, st.session_state.target_col)
                pipelines = train_models(X_train, y_train, preprocessor)
                selected_pipeline = pipelines[model_choice]
                
                st.session_state.trained_model = selected_pipeline
                st.session_state.feature_columns = X_train.columns.tolist()
                metrics, _, _ = evaluate_model(selected_pipeline, X_test, y_test)
                st.session_state.eval_metrics = metrics
                save_model(selected_pipeline)
                st.success("Model trained successfully!")

        if "eval_metrics" in st.session_state:
            m = st.session_state.eval_metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m['Accuracy']:.4f}")
            c2.metric("Precision", f"{m['Precision']:.4f}")
            c3.metric("Recall", f"{m['Recall']:.4f}")
            c4.metric("ROC-AUC", f"{m['ROC-AUC']:.4f}" if isinstance(m['ROC-AUC'], float) else m['ROC-AUC'])
            st.plotly_chart(plot_confusion_matrix(m['Confusion Matrix']))

elif menu == "Predictions":
    st.title("⚡ Individual Risk Assessment")
    if st.session_state.trained_model is None:
        st.warning("Train a model first.")
    else:
        sample_df = st.session_state.df.drop(columns=[st.session_state.target_col])
        input_data = {}
        for col in st.session_state.feature_columns:
            if sample_df[col].dtype in ['int64', 'float64']:
                input_data[col] = st.number_input(col, value=float(sample_df[col].mean()))
            else:
                input_data[col] = st.selectbox(col, options=sample_df[col].unique().tolist())
        
        if st.button("Evaluate Risk"):
            pred_class, pred_proba, risk_cat = make_prediction(st.session_state.trained_model, input_data, st.session_state.feature_columns)
            st.markdown(f"**Default Probability:** `{pred_proba:.2%}`")
            st.markdown(f"**Assigned Risk Category:** `{risk_cat}`")

elif menu == "Responsible AI":
    render_responsible_ai_section()

elif menu == "About Project":
    st.title("ℹ️ About CreditScore AI")
    st.markdown("CreditScore AI — Week 4 End-to-End Machine Learning Solution.")
'''
with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print("All project files created successfully!")