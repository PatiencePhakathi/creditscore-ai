# 💳 CreditScore AI: South African Credit Qualification Portal 🇿🇦

CreditScore AI is a machine learning-powered web application built to evaluate consumer credit qualification and default risk for the South African FinTech market. Aligned with local financial metrics and National Credit Regulator (NCR) guidelines, the app provides real-time credit scoring via structured inputs or conversational text narratives.

---

## 🌟 Key Features

* **📝 Dual Assessment Engine:**
  * **Structured SA Form:** Clear, localized form fields using ZAR (R) values, South African employment sectors, store accounts (e.g., Edgars, Woolworths, TFG), and banking attributes.
  * **🤖 AI Narrative Parser:** Allows applicants to type or paste a personal financial summary to automatically extract features and generate instant credit predictions.
* **🇿🇦 Fully Localized Context:** Tailored to South African affordability metrics and local credit criteria.
* **🔬 Interactive Model Lab:** Train and evaluate Random Forest classification models with confusion matrices, accuracy, precision, recall, and ROC-AUC scores.
* **🛡️ Responsible AI:** Embedded guidelines supporting fair lending practices aligned with the National Credit Act (NCA).

---

## 📂 Project Structure

```text
creditscore-ai/
│
├── app.py                   # Streamlit UI dashboard
├── data_loader.py           # SA synthetic dataset generator & CSV loader
├── preprocessing.py         # Feature engineering & scaling pipelines
├── model_training.py        # Random Forest & ML model training
├── evaluation.py            # Performance metrics & confusion matrix
├── prediction.py            # Real-time inference & risk tiering
├── visualizations.py        # Interactive Plotly chart functions
├── responsible_ai.py        # Fairness & compliance section
├── requirements.txt         # Project dependencies
└── tests/                   # Pytest suite
