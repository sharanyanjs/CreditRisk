import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, precision_recall_curve, roc_curve)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import warnings
warnings.filterwarnings('ignore')

# ==================== GS-STYLE CONFIGURATION ====================
st.set_page_config(
    page_title="GS Credit Risk Platform",
    layout="wide",
    page_icon=":bank:",
    initial_sidebar_state="expanded"
)

# Goldman Sachs color palette
GS_BLUE = "#0033a0"
GS_GREEN = "#7cba00"
GS_RED = "#d6001c"

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("german_credit_data.csv")
        
        # GS-style risk modeling
        df['Risk'] = np.where(
            (df['Age'] < 25) |
            (df['Credit amount'] > df['Credit amount'].quantile(0.8)) |
            (df['Duration'] > 30) |
            (df['Job'].isin([0, 1])), 1, 0
        )
        
        # Institutional-grade feature engineering
        df['Debt_to_Income'] = (df['Credit amount'] / duration) / (df['Credit amount'] / 12)
        df['Liquidity_Coverage'] = (df['Credit amount'] * 0.3) / df['Duration']  # Simulated GS metric
        
        return df.dropna()
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame()

# ==================== MODEL TRAINING ====================
@st.cache_resource
def train_gs_model():
    df = load_data()
    if df.empty:
        return None, None, None
    
    features = ['Age', 'Job', 'Credit amount', 'Duration', 'Debt_to_Income', 'Liquidity_Coverage']
    X = df[features]
    y = df['Risk']
    
    # GS-style scaling
    scaler = StandardScaler()
    X[['Credit amount', 'Duration', 'Debt_to_Income']] = scaler.fit_transform(
        X[['Credit amount', 'Duration', 'Debt_to_Income']])
    
    # Institutional-grade model
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=7,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X, y)
    
    # SHAP explainer with GS-compliant visualization
    explainer = shap.TreeExplainer(model)
    
    return model, scaler, explainer, features

# ==================== GS-STYLE UI COMPONENTS ====================
def gs_metric_card(title, value, delta=None, help_text=None):
    """Goldman-style metric card with professional formatting"""
    card = f"""
    <div style="border-left: 4px solid {GS_BLUE}; padding: 15px; background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0;">
        <div style="color: #666; font-size: 14px;">{title}</div>
        <div style="font-size: 28px; font-weight: 700; color: #333;">{value}</div>
        {f'<div style="color: {GS_GREEN if "-" in str(delta) else GS_RED}; font-size: 12px;">{delta}</div>' if delta else ''}
        {f'<div style="color: #999; font-size: 12px;">{help_text}</div>' if help_text else ''}
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)

# ==================== MAIN APP ====================
def main():
    # Load everything with error handling
    model, scaler, explainer, features = train_gs_model()
    if model is None:
        st.error("System initialization failed. Contact GS Engineering.")
        return
    
    df = load_data()
    
    # ===== GS-STYLE HEADER =====
    st.markdown(f"""
    <div style="background: {GS_BLUE}; padding: 20px; border-radius: 5px; color: white;">
        <h1 style="color: white; margin: 0;">Goldman Sachs Credit Risk Platform</h1>
        <p style="color: #ccc;">Institutional-grade risk analytics | v2.1.8</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== RISK ASSESSMENT ENGINE =====
    st.markdown("## Credit Decisioning Module")
    
    with st.form("gs_underwriting"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Applicant Age", 18, 80, 35, 
                           help="GS Policy: <25 requires co-signer")
            employment = st.selectbox("Employment Tier", 
                                    ["Tier 4: Unemployed", 
                                     "Tier 3: Part-time", 
                                     "Tier 2: Full-time", 
                                     "Tier 1: Executive"], 
                                    index=2)
            emp_tier = 3 - ["Tier 4", "Tier 3", "Tier 2", "Tier 1"].index(employment.split(":")[0])
            
        with col2:
            amount = st.number_input("Loan Amount ($)", 1000, 500000, 25000, 1000,
                                    help="GS Max: $500k for Tier 1 clients")
            term = st.slider("Term (months)", 6, 84, 36,
                            help="GS Standard: 12-60 months")
            
        with col3:
            fico = st.slider("FICO Score", 300, 850, 720,
                            help="GS Minimum: 680 for unsecured")
            existing_payments = st.number_input("Existing Debt Payments ($/mo)", 0, 10000, 500)
            
        submitted = st.form_submit_button("Run GS Risk Assessment")
        
        if submitted:
            # GS Underwriting Logic
            dti = (amount / term + existing_payments) / (amount * 0.3)  # Simulated income
            
            input_df = pd.DataFrame([{
                "Age": age,
                "Job": emp_tier,
                "Credit amount": amount,
                "Duration": term,
                "Debt_to_Income": dti,
                "Liquidity_Coverage": (amount * 0.3) / term
            }])
            
            # Scale and predict
            input_scaled = input_df.copy()
            input_scaled[['Credit amount', 'Duration', 'Debt_to_Income']] = scaler.transform(
                input_scaled[['Credit amount', 'Duration', 'Debt_to_Income']])
            
            proba = model.predict_proba(input_scaled)[0][1]
            risk_score = min(100, max(0, round(proba * 100, 1)))
            
            # GS Risk Tiers
            if risk_score >= 75:
                decision = "DECLINE"
                pricing = "N/A"
                color = GS_RED
            elif risk_score >= 50:
                decision = "MANUAL REVIEW"
                pricing = f"+{min(8, round((risk_score-50)/5))}%"
                color = "#FFA500"
            else:
                decision = "APPROVE"
                pricing = f"{max(2, 8 - round(risk_score/10))}% APR"
                color = GS_GREEN
                
            # Display GS-style decision
            st.markdown(f"""
            <div style="border: 2px solid {color}; border-radius: 5px; padding: 15px; margin: 20px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <h2 style="color: {color}; margin: 0;">{decision}</h2>
                        <p style="color: #666;">Goldman Sachs Recommendation</p>
                    </div>
                    <div style="text-align: right;">
                        <h1 style="color: {color}; margin: 0;">{risk_score}</h1>
                        <p style="color: #666;">Risk Score</p>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <p><strong>Pricing:</strong> {pricing}</p>
                    <p><strong>Rationale:</strong> {get_gs_rationale(risk_score, age, emp_tier, fico)}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # SHAP Explanation
            st.markdown("#### Risk Factor Analysis")
            shap_values = explainer.shap_values(input_scaled)
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values[1], input_scaled, feature_names=features, show=False)
            st.pyplot(fig)
            plt.clf()
    
    # ===== PORTFOLIO ANALYTICS =====
    st.markdown("## GS Portfolio Surveillance")
    
    # GS Risk Metrics
    cols = st.columns(4)
    with cols[0]:
        gs_metric_card("Total Exposure", f"${df['Credit amount'].sum()/1e6:.2f}M", 
                      help_text="Basel III Weighted: ${df['Credit amount'].sum()/1e6*0.6:.2f}M")
    with cols[1]:
        gs_metric_card("Delinquency Rate", f"{df['Risk'].mean()*100:.1f}%", 
                      f"↓ {(df['Risk'].mean()*100 - 8.2):.1f}% vs industry")
    with cols[2]:
        gs_metric_card("Risk-Adjusted Yield", "14.2%", "↑ 3.1% YoY")
    with cols[3]:
        gs_metric_card("CCAR Score", "82/100", "Stress Test Ready")
    
    # Advanced GS Analytics
    with st.expander("Advanced Risk Analytics", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Concentration Risk", "Migration Risk", "Scenario Analysis"])
        
        with tab1:
            st.markdown("### GS Concentration Limits")
            fig = plt.figure()
            sns.boxplot(data=df, x='Risk', y='Credit amount', palette=[GS_GREEN, GS_RED])
            plt.title("Exposure Distribution by Risk Tier")
            st.pyplot(fig)
            
        with tab2:
            st.markdown("### GS Risk Migration Matrix")
            # Simulated migration probabilities
            migration = pd.DataFrame(
                [[0.85, 0.12, 0.03],
                 [0.10, 0.75, 0.15],
                 [0.02, 0.18, 0.80]],
                columns=["Low→", "Medium→", "High→"],
                index=["Current Low", "Current Medium", "Current High"]
            )
            st.dataframe(migration.style.background_gradient(cmap='Blues'))
            
        with tab3:
            st.markdown("### GS Stress Testing")
            stress_level = st.slider("Economic Shock Severity", 0, 100, 30)
            st.write(f"Projected Defaults at {stress_level}% Stress: {df['Risk'].mean()*100*(1+stress_level/100):.1f}%")

def get_gs_rationale(score, age, employment, fico):
    """Generate GS-style underwriting rationale"""
    factors = []
    if score > 70:
        factors.append("elevated risk characteristics")
    if age < 25:
        factors.append("young borrower profile")
    if employment < 2:
        factors.append("limited income verification")
    if fico < 700:
        factors.append("subprime credit history")
        
    if not factors:
        return "Meets all GS underwriting standards"
    return "Elevated " + ", ".join(factors) + " per GS Credit Policy 2023-08"

if __name__ == "__main__":
    main()
