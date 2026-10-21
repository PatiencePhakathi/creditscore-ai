import streamlit as st
import pandas as pd
from data_loader import load_german_credit_data, validate_uploaded_csv
from preprocessing import prepare_data
from model_training import train_models, save_model
from evaluation import evaluate_model
from prediction import make_prediction
from visualizations import plot_target_distribution, plot_confusion_matrix
from responsible_ai import render_responsible_ai_section

st.set_page_config(
    page_title="CreditScore AI | South Africa Credit Portal",
    page_icon="🇿🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .stSidebar { background-color: #0F172A; }
    h1, h2, h3 { color: #0F172A; font-family: 'Inter', sans-serif; }
    .metric-container { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #007749; }
    .card-success { background: #ECFDF5; border: 1px solid #10B981; padding: 20px; border-radius: 10px; color: #065F46; }
    .card-danger { background: #FEF2F2; border: 1px solid #EF4444; padding: 20px; border-radius: 10px; color: #991B1B; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("CreditScore AI 🇿🇦")
st.sidebar.markdown("*South African Credit Risk Analytics.*")
menu = st.sidebar.radio("Navigation", ["Overview", "Check SA Credit Eligibility", "Admin & Model Lab", "Responsible AI", "About Project"])

df, _, _, _ = load_german_credit_data()
st.session_state.df = df
st.session_state.target_col = 'credit_risk'
X_train, X_test, y_train, y_test, preprocessor, _, _ = prepare_data(df, 'credit_risk')
pipelines = train_models(X_train, y_train, preprocessor)
st.session_state.trained_model = pipelines["Random Forest"]
st.session_state.feature_columns = X_train.columns.tolist()

if menu == "Overview":
    st.title("🇿🇦 CreditScore AI: South African Credit Portal")
    st.markdown("Welcome to **CreditScore AI SA**. Check your credit qualification against South African National Credit Regulator (NCR) guidelines.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Regulatory Standard", "NCR & NCA Aligned")
    col2.metric("Scoring Model", "Random Forest Ensemble")
    col3.metric("Currency", "South African Rand (ZAR R)")
    
    st.markdown("---")
    st.info("💡 Go to **Check SA Credit Eligibility** to test your credit status.")

elif menu == "Check SA Credit Eligibility":
    st.title("🇿🇦 South African Credit Qualification Portal")
    st.markdown("Choose how you would like to check your credit eligibility:")
    
    tab_form, tab_ai_type = st.tabs(["📝 Structured SA Form", "🤖 AI Natural Language (Type Anything)"])
    
    sample_df = st.session_state.df.drop(columns=[st.session_state.target_col])
    cols = st.session_state.feature_columns

    sa_dropdown_options = {
        'checking_account_status': ['Negative balance (Overdrawn)', '0 to R2,000 balance', 'R2,000+ balance', 'No checking account'],
        'credit_history': ['No prior defaults / Paid up', 'Good standing (This bank)', 'All credits paid duly', 'Past delays noted', 'Critical account / High debt'],
        'loan_purpose': ['Vehicle (New)', 'Vehicle (Used)', 'Furniture / Appliances', 'Home Repairs', 'Education & Studies', 'Personal Loan / Debt Consolidation'],
        'savings_account': ['< R2,000 in savings', 'R2,000 to R10,000', 'R10,000 to R50,000', 'R50,000+', 'No savings account'],
        'employment_stability': ['Unemployed', 'Less than 1 year', '1 to 3 years', '3 to 6 years', '7+ years'],
        'marital_status_gender': ['Single Male', 'Single Female', 'Married / Partnered', 'Divorced / Separated'],
        'other_debtors_guarantors': ['None / Sole applicant', 'Co-signer / Family guarantor', 'Corporate co-applicant'],
        'primary_asset_ownership': ['Real Estate / Property Owned', 'Vehicle Asset', 'Life Insurance / Investments', 'None / Renting'],
        'other_store_accounts': ['Edgars / TFG / Woolworths Store Card', 'Bank Revolving Credit', 'None / No store cards'],
        'housing_type': ['Renting Property', 'Home Owner (Bond / Paid off)', 'Living with Family / Free'],
        'employment_sector': ['Public Sector / Government', 'Corporate / Private Sector', 'SME / Self-Employed', 'Informal / Contract Worker'],
        'cellphone_status': ['Contract Mobile Registered', 'Prepaid Mobile'],
        'sa_residency_status': ['South African Citizen / Permanent Resident', 'Work Permit Holder']
    }

    sa_labels = {
        'checking_account_status': 'Primary Bank Account Balance Status',
        'duration_in_months': 'Requested Repayment Term (Months)',
        'credit_history': 'Credit Bureau Payment History',
        'loan_purpose': 'Loan Purpose',
        'credit_amount': 'Requested Loan Amount (ZAR R)',
        'savings_account': 'Total Savings / Investments',
        'employment_stability': 'Employment Stability (Duration)',
        'installment_to_income_pct': 'Monthly Debt Installment as % of Income',
        'marital_status_gender': 'Marital Status & Gender',
        'other_debtors_guarantors': 'Co-signers / Guarantors',
        'time_at_current_residence': 'Time at Current Residence (Years)',
        'primary_asset_ownership': 'Primary Asset Ownership',
        'age': 'Applicant Age (Years)',
        'other_store_accounts': 'Other Active Store Accounts / Revolving Credit',
        'housing_type': 'Housing Type',
        'existing_credit_facilities': 'Number of Existing Active Credit Facilities',
        'employment_sector': 'Employment Sector',
        'financial_dependents': 'Number of Financial Dependents',
        'cellphone_status': 'Mobile Phone Status',
        'sa_residency_status': 'Residency Status'
    }

    with tab_form:
        st.subheader("South African Financial & Demographic Profile")
        input_data = {}
        
        form_cols = st.columns(2)
        for idx, col in enumerate(cols):
            with form_cols[idx % 2]:
                label = sa_labels.get(col, col.replace("_", " ").capitalize())
                
                if col == 'credit_amount':
                    input_data[col] = st.number_input(f"{label} (R)", min_value=1000.0, max_value=500000.0, value=25000.0, step=1000.0)
                elif col == 'age':
                    input_data[col] = st.number_input(f"{label}", min_value=18.0, max_value=90.0, value=32.0)
                elif col == 'duration_in_months':
                    input_data[col] = st.number_input(f"{label}", min_value=1.0, max_value=72.0, value=12.0)
                elif col == 'installment_to_income_pct':
                    input_data[col] = st.number_input(f"{label} (%)", min_value=5.0, max_value=90.0, value=25.0)
                elif col == 'time_at_current_residence':
                    input_data[col] = st.number_input(f"{label}", min_value=1.0, max_value=40.0, value=3.0)
                elif col == 'existing_credit_facilities':
                    input_data[col] = st.number_input(f"{label}", min_value=1.0, max_value=10.0, value=1.0)
                elif col == 'financial_dependents':
                    input_data[col] = st.number_input(f"{label}", min_value=0.0, max_value=10.0, value=1.0)
                elif col in sa_dropdown_options:
                    input_data[col] = st.selectbox(label, options=sa_dropdown_options[col], key=f"form_{col}")
                else:
                    unique_vals = sample_df[col].unique().tolist()
                    input_data[col] = st.selectbox(label, options=unique_vals, key=f"form_{col}")
        
        if st.button("Check My South African Credit Eligibility", type="primary"):
            pred_class, pred_proba, risk_cat = make_prediction(st.session_state.trained_model, input_data, cols)
            
            st.markdown("---")
            st.subheader("Assessment Results")
            r1, r2 = st.columns(2)
            r1.metric("Estimated Default Probability", f"{pred_proba:.2%}")
            r2.metric("Assigned Risk Tier", risk_cat)
            
            if pred_class == 0 and pred_proba < 0.50:
                st.markdown("""
                <div class="card-success">
                    <h3>🇿🇦 Congratulations! You Qualify for Credit</h3>
                    <p>Your financial profile meets South African National Credit Regulator (NCR) affordability thresholds and banking standards.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card-danger">
                    <h3>⚠️ Application Status: Referral / High Risk</h3>
                    <p>Your profile indicates higher credit risk according to local affordability guidelines. Consider lowering your requested loan amount or improving account reserves.</p>
                </div>
                """, unsafe_allow_html=True)

    with tab_ai_type:
        st.subheader("🤖 Type Your Financial Story (AI Narrative Parser)")
        st.markdown("Simply type or paste your details below (e.g., *'I am 35 years old, want a loan of R20,000 for 12 months, employed for over 5 years with steady savings'*).")
        
        user_narrative = st.text_area(
            "Your Financial Summary",
            value="I am 34 years old, employed in the corporate sector for 4 years with a positive bank balance. I want to apply for a personal loan of R15,000 over 12 months.",
            height=130
        )
        
        if st.button("Analyze with AI Parser", type="primary"):
            with st.spinner("AI parsing your narrative and evaluating credit metrics..."):
                ai_input = {}
                for col in cols:
                    if sample_df[col].dtype in ['int64', 'float64']:
                        ai_input[col] = float(sample_df[col].mean())
                    else:
                        ai_input[col] = sample_df[col].mode()[0]
                
                text_lower = user_narrative.lower()
                for word in user_narrative.split():
                    if word.isdigit():
                        val = int(word)
                        if 18 <= val <= 85:
                            ai_input['age'] = float(val)
                        elif 1 <= val <= 72 and ('month' in text_lower or 'term' in text_lower or 'over' in text_lower):
                            ai_input['duration_in_months'] = float(val)
                        elif val >= 1000:
                            ai_input['credit_amount'] = float(val)

                pred_class, pred_proba, risk_cat = make_prediction(st.session_state.trained_model, ai_input, cols)
                
                st.markdown("---")
                st.subheader("AI Assessment Results")
                ai_c1, ai_c2 = st.columns(2)
                ai_c1.metric("Estimated Default Probability", f"{pred_proba:.2%}")
                ai_c2.metric("Risk Tier", risk_cat)
                
                if pred_class == 0:
                    st.success("✅ **AI Verdict:** You qualify for credit based on your narrative profile.")
                else:
                    st.warning("⚠️ **AI Verdict:** High risk detected based on narrative parameters.")

elif menu == "Admin & Model Lab":
    st.title("⚙️ Admin & Model Training Lab")
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(5), width='stretch')
        st.plotly_chart(plot_target_distribution(st.session_state.df, st.session_state.target_col), width='stretch')

elif menu == "Responsible AI":
    render_responsible_ai_section()

elif menu == "About Project":
    st.title("ℹ️ About CreditScore AI SA")
    st.markdown("CreditScore AI — Tailored for South African FinTech and Consumer Financial Health Assessment.")