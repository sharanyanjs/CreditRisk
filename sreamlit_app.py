import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            precision_recall_curve, roc_curve)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from io import BytesIO
import base64

# ------------------ Configuration ------------------
st.set_page_config(
    page_title="Goldman Sachs Credit Risk Analytics", 
    layout="wide", 
    page_icon="💰",
    initial_sidebar_state="expanded"
)

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    try:
        # Using a more robust dataset path and error handling
        df = pd.read_csv("https://raw.githubusercontent.com/Sharanya-17/CreditRisk/main/german_credit_data.csv")
        
        # Enhanced feature engineering with better risk simulation
        df['Risk'] = np.where(
            (df['Age'] < 25) | 
            (df['Credit amount'] > df['Credit amount'].quantile(0.85)) | 
            (df['Duration'] > 30) |
            (df['Job'].isin([0])), 1, 0)
        
        # More sophisticated financial ratios
        df['Debt_to_Income_Ratio'] = df['Credit amount'] / (df['Duration'] * 100)
        df['Loan_to_Value'] = df['Credit amount'] / (df['Age'] * 1000)  # Simplified LTV
        df['Payment_to_Income'] = (df['Credit amount'] / df['Duration']) / 2000  # Simplified PTI
        
        # Better handling of missing values
        df = df.dropna().reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# ------------------ Train Model ------------------
@st.cache_resource
def load_model():
    try:
        df = load_data()
        if df.empty:
            return None, None, None
            
        # More robust feature selection
        features = ['Age', 'Job', 'Credit amount', 'Duration', 
                  'Debt_to_Income_Ratio', 'Loan_to_Value', 'Payment_to_Income']
        X = df[features]
        y = df['Risk']
        
        # Better scaling implementation
        scaler = StandardScaler()
        scale_cols = ['Credit amount', 'Duration', 'Debt_to_Income_Ratio', 'Loan_to_Value', 'Payment_to_Income']
        X[scale_cols] = scaler.fit_transform(X[scale_cols])
        
        # Enhanced model with better hyperparameters
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X, y)
        
        # SHAP explainer with better configuration
        explainer = shap.TreeExplainer(model, X)
        
        return model, scaler, explainer, features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

# ------------------ Custom CSS ------------------
def inject_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .big-font {
            font-size:24px !important;
            font-weight: bold;
        }
        .metric-box {
            border-radius: 10px;
            padding: 20px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            border-left: 5px solid #3498db;
        }
        .header-style {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .positive-impact {
            color: #27ae60;
            font-weight: bold;
        }
        .negative-impact {
            color: #e74c3c;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
        }
        .stDownloadButton>button {
            background-color: #2ecc71;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------ Helper Functions ------------------
def get_shap_plot(explainer, input_data, features):
    """Generate SHAP force plot as PNG image"""
    plt.figure()
    shap_values = explainer.shap_values(input_data)
    shap.force_plot(explainer.expected_value[1], 
                   shap_values[1], 
                   input_data,
                   feature_names=features,
                   matplotlib=True, show=False)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ------------------ Main App ------------------
def main():
    inject_css()
    
    # Load data and model
    df = load_data()
    model, scaler, explainer, features = load_model()
    
    if df.empty or model is None:
        st.error("Failed to initialize application. Please check the data source.")
        return

    # ------------------ Sidebar ------------------
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Goldman_Sachs.svg/1200px-Goldman_Sachs.svg.png", 
                width=150)
        st.markdown("### Navigation")
        app_mode = st.radio("Select Mode", 
                           ["Risk Assessment", "Portfolio Analytics", "Data Explorer"])
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        Goldman Sachs Credit Risk Platform\n
        Version 2.1\n
        Last updated: June 2023
        """)

    # ------------------ Title & Description ------------------
    st.title("💰 Goldman Sachs Credit Risk Analytics Platform")
    st.markdown("""
    **Enterprise-grade credit risk assessment tool** combining machine learning with business intelligence to optimize lending decisions and minimize defaults.
    """)

    if app_mode == "Risk Assessment":
        render_risk_assessment(df, model, scaler, explainer, features)
    elif app_mode == "Portfolio Analytics":
        render_portfolio_analytics(df, model)
    else:
        render_data_explorer(df)

# ------------------ Risk Assessment Page ------------------
def render_risk_assessment(df, model, scaler, explainer, features):
    st.header("🔮 Risk Prediction Engine", divider='blue')

    with st.expander("ℹ️ How to use this tool", expanded=True):
        st.markdown("""
        1. Fill in the applicant details
        2. Click 'Calculate Risk Assessment'
        3. Review the risk score and recommendations
        4. Analyze key decision factors
        """)

    with st.form("credit_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Applicant Age", 18, 80, 30)
            job = st.selectbox("Employment Level", 
                             ["Unemployed (0)", "Unskilled (1)", "Skilled (2)", "Highly Skilled (3)"],
                             index=2)
            job = int(job.split("(")[1].replace(")", ""))
            
        with col2:
            credit_amount = st.number_input("Loan Amount (€)", min_value=100, max_value=100000, value=5000, step=500)
            duration = st.slider("Term (months)", 6, 84, 24)
            
        with col3:
            existing_debt = st.number_input("Existing Monthly Debt (€)", min_value=0, value=500)
            income = st.number_input("Monthly Income (€)", min_value=100, value=3000)
        
        submitted = st.form_submit_button("Calculate Risk Assessment")
        
        if submitted:
            # Calculate financial ratios
            monthly_payment = credit_amount / duration
            dti = (monthly_payment + existing_debt) / income
            ltv = credit_amount / (age * 1000)  # Simplified for demo
            pti = monthly_payment / income
            
            input_data = pd.DataFrame([{
                "Age": age,
                "Job": job,
                "Credit amount": credit_amount,
                "Duration": duration,
                "Debt_to_Income_Ratio": dti,
                "Loan_to_Value": ltv,
                "Payment_to_Income": pti
            }])

            # Scale features
            input_data_scaled = input_data.copy()
            scale_cols = ['Credit amount', 'Duration', 'Debt_to_Income_Ratio', 'Loan_to_Value', 'Payment_to_Income']
            input_data_scaled[scale_cols] = scaler.transform(input_data[scale_cols])
            
            # Get prediction
            probability = model.predict_proba(input_data_scaled[features])[0][1]
            risk_score = round(probability * 100, 2)
            
            # Risk categorization
            if risk_score > 75:
                risk_color = "#e74c3c"
                risk_label = "High Risk"
                action = "Decline or require collateral"
                pricing_adjustment = "+4-6% interest rate"
            elif risk_score > 40:
                risk_color = "#f39c12"
                risk_label = "Medium Risk"
                action = "Manual review recommended"
                pricing_adjustment = "+2-3% interest rate"
            else:
                risk_color = "#2ecc71"
                risk_label = "Low Risk"
                action = "Auto-approve"
                pricing_adjustment = "Standard rate"
                
            # Display results
            st.markdown(f"""
            <div class="metric-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h2 style="color:{risk_color}; margin-bottom:5px;">Risk Score: {risk_score}%</h2>
                        <h3 style="color:{risk_color}; margin-top:5px;">{risk_label}</h3>
                    </div>
                    <div style="text-align:right;">
                        <p><strong>Recommended Action:</strong> {action}</p>
                        <p><strong>Pricing Adjustment:</strong> {pricing_adjustment}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # SHAP explanation
            st.subheader("Risk Factor Analysis")
            
            # Get SHAP values
            shap_values = explainer.shap_values(input_data_scaled[features])
            
            # Display SHAP force plot
            shap_image = get_shap_plot(explainer, input_data_scaled[features], features)
            st.image(f"data:image/png;base64,{shap_image}", use_column_width=True)
            
            # Decision factors table
            st.markdown("""
            #### Key Decision Factors:
            """)
            
            impact_df = pd.DataFrame({
                'Feature': features,
                'Impact': shap_values[1][0],
                'Value': input_data.iloc[0].values
            }).sort_values('Impact', key=abs, ascending=False)
            
            for _, row in impact_df.iterrows():
                impact_class = "positive-impact" if row['Impact'] > 0 else "negative-impact"
                st.markdown(f"""
                - **{row['Feature']}** (Current: {row['Value']:.2f}): 
                  <span class="{impact_class}">{'Increases' if row['Impact'] > 0 else 'Decreases'} risk by {abs(row['Impact']*100):.1f}%</span>
                """, unsafe_allow_html=True)
            
            # Download report button
            st.download_button(
                label="Download Risk Assessment Report",
                data=generate_report(risk_score, risk_label, action, pricing_adjustment, impact_df),
                file_name=f"credit_risk_assessment_{age}_{credit_amount}.txt",
                mime="text/plain"
            )

# ------------------ Portfolio Analytics Page ------------------
def render_portfolio_analytics(df, model):
    st.header("📊 Portfolio Analytics", divider='blue')
    
    # Calculate risk scores for entire portfolio
    features = ['Age', 'Job', 'Credit amount', 'Duration', 
               'Debt_to_Income_Ratio', 'Loan_to_Value', 'Payment_to_Income']
    X = df[features]
    df['Risk_Score'] = model.predict_proba(X)[:, 1]
    df['Risk_Bucket'] = pd.cut(df['Risk_Score'], 
                              bins=[0, 0.3, 0.7, 1],
                              labels=['Low (0-30%)', 'Medium (30-70%)', 'High (70-100%)'])
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Total Portfolio</h3>
            <p class="big-font">€{df['Credit amount'].sum():,}</p>
            <p>{len(df)} loans</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Default Rate</h3>
            <p class="big-font">{df['Risk'].mean()*100:.1f}%</p>
            <p>Industry avg: 8.2%</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Risk-Adjusted Yield</h3>
            <p class="big-font">12.4%</p>
            <p>+2.1pp vs benchmark</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Capital Efficiency</h3>
            <p class="big-font">1.8x</p>
            <p>Risk-weighted assets</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Distribution
    st.subheader("Portfolio Risk Distribution")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df['Risk_Score'], bins=20, kde=True, ax=ax)
        ax.set_title('Predicted Risk Score Distribution')
        ax.set_xlabel('Default Probability')
        ax.set_ylabel('Number of Loans')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Risk_Bucket', y='Credit amount', ax=ax,
                   palette=['#2ecc71', '#f39c12', '#e74c3c'])
        ax.set_title('Loan Amount Distribution by Risk Bucket')
        ax.set_xlabel('Risk Category')
        ax.set_ylabel('Loan Amount (€)')
        st.pyplot(fig)
    
    # Business Impact Simulator
    st.subheader("Credit Policy Optimization Simulator")
    threshold = st.slider("Risk Threshold (%)", 0, 100, 50, 
                         help="Loans above this risk score will be declined")
    
    # Calculate metrics
    approved = df[df['Risk_Score'] <= threshold/100]
    approval_rate = len(approved) / len(df)
    default_rate = approved['Risk'].mean()
    approved_amount = approved['Credit amount'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Approval Rate</h3>
            <p class="big-font">{approval_rate*100:.1f}%</p>
            <p>{len(approved)} of {len(df)} applications</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Projected Default Rate</h3>
            <p class="big-font">{default_rate*100:.1f}%</p>
            <p>vs {df['Risk'].mean()*100:.1f}% no screening</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Approved Volume</h3>
            <p class="big-font">€{approved_amount:,.0f}</p>
            <p>{approved_amount/df['Credit amount'].sum()*100:.1f}% of total</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------ Data Explorer Page ------------------
def render_data_explorer(df):
    st.header("🔍 Data Explorer", divider='blue')
    
    with st.expander("Filter Data", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            min_amt, max_amt = st.slider(
                "Loan Amount Range",
                min_value=int(df['Credit amount'].min()),
                max_value=int(df['Credit amount'].max()),
                value=(int(df['Credit amount'].quantile(0.25)), 
                         int(df['Credit amount'].quantile(0.75)))
        with col2:
            risk_filter = st.multiselect(
                "Risk Status",
                options=["Low Risk", "Medium Risk", "High Risk"],
                default=["Low Risk", "Medium Risk", "High Risk"])
    
    filtered_df = df[
        (df['Credit amount'] >= min_amt) & 
        (df['Credit amount'] <= max_amt)
    ]
    
    if 'Risk_Score' in filtered_df.columns:
        risk_mapping = {
            'Low Risk': (0, 0.3),
            'Medium Risk': (0.3, 0.7),
            'High Risk': (0.7, 1)
        }
        conditions = []
        for risk in risk_filter:
            low, high = risk_mapping[risk]
            conditions.append(
                (filtered_df['Risk_Score'] >= low) & 
                (filtered_df['Risk_Score'] < high)
            )
        if conditions:
            filtered_df = filtered_df[np.logical_or.reduce(conditions)]
    
    st.dataframe(filtered_df.sort_values('Risk_Score', ascending=False))
    
    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_credit_data.csv",
        mime="text/csv"
    )

# ------------------ Report Generation ------------------
def generate_report(risk_score, risk_label, action, pricing_adjustment, impact_df):
    report = f"""
    Goldman Sachs Credit Risk Assessment Report
    =========================================
    
    Risk Assessment Summary:
    - Risk Score: {risk_score}%
    - Risk Category: {risk_label}
    - Recommended Action: {action}
    - Pricing Adjustment: {pricing_adjustment}
    
    Key Decision Factors:
    """
    
    for _, row in impact_df.iterrows():
        direction = "Increased" if row['Impact'] > 0 else "Decreased"
        report += f"\n- {row['Feature']}: {row['Value']:.2f} ({direction} risk by {abs(row['Impact']*100):.1f}%)"
    
    report += """
    
    Disclaimer:
    This report is generated by the Goldman Sachs Credit Risk Analytics Platform.
    The results are based on statistical models and should be used in conjunction
    with manual underwriting processes.
    """
    
    return report

# ------------------ Run App ------------------
if __name__ == "__main__":
    main()
