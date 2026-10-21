import streamlit as st

def render_responsible_ai_section():
    st.subheader("Responsible AI, Fairness & Governance")
    st.markdown("""
    In financial services, automated decision-making carries profound ethical and regulatory obligations. 
    CreditScore AI integrates rigorous governance principles:
    * **Historical & Sampling Bias:** Models trained on historical lending data may reflect past systemic exclusions.
    * **False Positives vs. False Negatives:** Rejecting creditworthy applicants vs. approving high default risk.
    * **Human Oversight:** Predictions are analytical estimates and must never replace professional underwriting judgment.
    """)
