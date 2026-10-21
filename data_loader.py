import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(show_spinner=True)
def load_german_credit_data():
    """
    Generates a realistic South African Credit Risk dataset aligned with 
    NCR (National Credit Regulator) and local retail banking credit assessment criteria.
    """
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'checking_account_status': np.random.choice(['Negative balance (Overdrawn)', '0 to R2,000 balance', 'R2,000+ balance', 'No checking account'], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
        'duration_in_months': np.random.choice([6, 12, 18, 24, 36, 48, 60], n_samples),
        'credit_history': np.random.choice(['No prior defaults / Paid up', 'Good standing (This bank)', 'All credits paid duly', 'Past delays noted', 'Critical account / High debt'], n_samples, p=[0.1, 0.3, 0.3, 0.2, 0.1]),
        'loan_purpose': np.random.choice(['Vehicle (New)', 'Vehicle (Used)', 'Furniture / Appliances', 'Home Repairs', 'Education & Studies', 'Personal Loan / Debt Consolidation'], n_samples),
        'credit_amount': np.random.randint(5000, 250000, n_samples),
        'savings_account': np.random.choice(['< R2,000 in savings', 'R2,000 to R10,000', 'R10,000 to R50,000', 'R50,000+', 'No savings account'], n_samples),
        'employment_stability': np.random.choice(['Unemployed', 'Less than 1 year', '1 to 3 years', '3 to 6 years', '7+ years'], n_samples, p=[0.1, 0.15, 0.3, 0.25, 0.2]),
        'installment_to_income_pct': np.random.randint(10, 60, n_samples),
        'marital_status_gender': np.random.choice(['Single Male', 'Single Female', 'Married / Partnered', 'Divorced / Separated'], n_samples),
        'other_debtors_guarantors': np.random.choice(['None / Sole applicant', 'Co-signer / Family guarantor', 'Corporate co-applicant'], n_samples, p=[0.8, 0.15, 0.05]),
        'time_at_current_residence': np.random.randint(1, 15, n_samples),
        'primary_asset_ownership': np.random.choice(['Real Estate / Property Owned', 'Vehicle Asset', 'Life Insurance / Investments', 'None / Renting'], n_samples),
        'age': np.random.randint(21, 68, n_samples),
        'other_store_accounts': np.random.choice(['Edgars / TFG / Woolworths Store Card', 'Bank Revolving Credit', 'None / No store cards'], n_samples),
        'housing_type': np.random.choice(['Renting Property', 'Home Owner (Bond / Paid off)', 'Living with Family / Free'], n_samples),
        'existing_credit_facilities': np.random.randint(1, 5, n_samples),
        'employment_sector': np.random.choice(['Public Sector / Government', 'Corporate / Private Sector', 'SME / Self-Employed', 'Informal / Contract Worker'], n_samples),
        'financial_dependents': np.random.randint(0, 5, n_samples),
        'cellphone_status': np.random.choice(['Contract Mobile Registered', 'Prepaid Mobile'], n_samples),
        'sa_residency_status': np.random.choice(['South African Citizen / Permanent Resident', 'Work Permit Holder'], n_samples, p=[0.95, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    risk_score = (
        (df['checking_account_status'] == 'Negative balance (Overdrawn)').astype(int) * 0.4 +
        (df['installment_to_income_pct'] > 45).astype(int) * 0.4 +
        (df['credit_history'] == 'Critical account / High debt').astype(int) * 0.5 +
        (df['employment_stability'] == 'Unemployed').astype(int) * 0.6 -
        (df['savings_account'] == 'R50,000+').astype(int) * 0.3 -
        (df['employment_stability'] == '7+ years').astype(int) * 0.2
    )
    df['credit_risk'] = (risk_score + np.random.normal(0, 0.2, n_samples) > 0.25).astype(int)
    
    return df, "South African Consumer Credit Risk Dataset (NCR Aligned)", "Local Synthetic Open-Data Repository", "CreditScore AI SA Analytics"

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