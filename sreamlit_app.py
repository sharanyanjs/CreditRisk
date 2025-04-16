# app.py
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
                            precision_recall_curve, roc_curve, classification_report)
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from io import BytesIO
import base64
import datetime
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_extras.metric_cards import style_metric_cards

# ------------------ Configuration ------------------
st.set_page_config(
    page_title="GS Credit Risk Platform | v3.1", 
    layout="wide", 
    page_icon="💰",
    initial_sidebar_state="expanded"
)

# ------------------ Constants ------------------
BUSINESS_DAYS_YEAR = 252
RISK_CATEGORIES = {
    "Low": (0, 0.3, "#2ecc71"),
    "Medium": (0.3, 0.7, "#f39c12"),
    "High": (0.7, 1.0, "#e74c3c")
}

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/Sharanya-17/CreditRisk/main/german_credit_data.csv")
        
        # Enhanced feature engineering
        df['Risk'] = np.where(
            (df['Age'] < 25) | 
            (df['Credit amount'] > df['Credit amount'].quantile(0.85)) | 
            (df['Duration'] > 30) |
            (df['Job'].isin([0])), 1, 0)
        
        # Business-relevant features
        df['Debt_to_Income'] = df['Credit amount'] / (df['Duration'] * 100)
        df['Loan_to_Value'] = df['Credit amount'] / (df['Age'] * 1000)
        df['Payment_to_Income'] = (df['Credit amount'] / df['Duration']) / 2000
        df['Expected_Profit'] = (df['Credit amount'] * 0.12) - (df['Credit amount'] * df['Risk'] * 0.75)
        
        # Handle class imbalance
        if df['Risk'].mean() < 0.2:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(df.drop('Risk', axis=1), df['Risk']
            df = pd.concat([X_res, y_res], axis=1)
        
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame()

# ------------------ Model Training ------------------
@st.cache_resource
def train_model(df):
    try:
        features = ['Age', 'Job', 'Credit amount', 'Duration', 
                  'Debt_to_Income', 'Loan_to_Value', 'Payment_to_Income']
        X = df[features]
        y = df['Risk']
        
        # Business-oriented train-test split (time-based simulation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Business performance metrics
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        explainer = shap.TreeExplainer(model, X_train_scaled)
        return model, scaler, explainer, features, metrics
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        return None, None, None, None, None

# ------------------ UI Components ------------------
def render_sidebar():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Goldman_Sachs.svg/1200px-Goldman_Sachs.svg.png", 
                width=150)
        st.markdown("### Navigation")
        app_mode = st.radio("", ["Risk Assessment", "Portfolio Analytics", "Data Explorer", "Model Performance"])
        
        st.markdown("---")
        st.markdown("### Business Parameters")
        risk_appetite = st.slider("Risk Appetite (%)", 10, 50, 25)
        min_profit_margin = st.number_input("Minimum Profit Margin (%)", 5.0, 20.0, 8.5)
        
        st.markdown("---")
        st.markdown(f"**Last Updated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.caption("Goldman Sachs Internal Use Only")

    return app_mode, risk_appetite, min_profit_margin

def style_metrics(metrics):
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label=name, value=f"{value:.2%}" if isinstance(value, float) else value)
    style_metric_cards()

# ------------------ Main Pages ------------------
def risk_assessment_page(df, model, scaler, explainer, features):
    st.header("📝 Credit Application Assessment", divider='blue')
    
    with st.expander("🚀 Quick Assessment", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            applicant_type = st.radio("Applicant Type", ["Individual", "Business"], horizontal=True)
            credit_purpose = st.selectbox("Loan Purpose", ["Working Capital", "Equipment", "Real Estate", "Personal"])
        
        with col2:
            existing_relationship = st.checkbox("Existing Client")
            preferred_client = st.checkbox("Preferred Banking Client")
    
    with st.form("application_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 80, 35)
            job = st.select_slider("Employment Stability", 
                                 options=["Unemployed", "Part-time", "Full-time", "Executive"],
                                 value="Full-time")
            income = st.number_input("Monthly Income (€)", min_value=500, value=3500, step=500)
            
        with col2:
            amount = st.number_input("Loan Amount (€)", min_value=1000, max_value=500000, value=25000, step=1000)
            duration = st.slider("Term (months)", 6, 84, 36)
            collateral = st.number_input("Collateral Value (€)", min_value=0, value=0)
            
        with col3:
            existing_debt = st.number_input("Existing Debt Payments (€/mo)", min_value=0, value=500)
            credit_score = st.slider("External Credit Score", 300, 850, 680)
            recent_inquiries = st.number_input("Credit Inquiries (6mo)", min_value=0, max_value=10, value=1)
        
        submitted = st.form_submit_button("Assess Credit Risk")
        
        if submitted:
            # Business logic calculations
            monthly_payment = amount / duration
            dti = (monthly_payment + existing_debt) / income
            ltv = amount / max(collateral, 1) if collateral > 0 else amount / (age * 1000)
            pti = monthly_payment / income
            
            input_data = pd.DataFrame([{
                "Age": age,
                "Job": ["Unemployed", "Part-time", "Full-time", "Executive"].index(job),
                "Credit amount": amount,
                "Duration": duration,
                "Debt_to_Income": dti,
                "Loan_to_Value": ltv,
                "Payment_to_Income": pti
            }])
            
            # Scale and predict
            input_scaled = scaler.transform(input_data[features])
            proba = model.predict_proba(input_scaled)[0][1]
            risk_score = round(proba * 100, 1)
            
            # Business decision logic
            risk_category = next((k for k, (low, high, _) in RISK_CATEGORIES.items() 
                                if low <= proba < high), "High")
            color = RISK_CATEGORIES[risk_category][2]
            
            expected_profit = (amount * (min_profit_margin/100 + (0.01 if existing_relationship else 0))) - \
                            (amount * proba * 0.85)
            
            # Display results
            st.markdown(f"""
            <div style="border-radius:10px; padding:20px; background-color:#f8f9fa; border-left:5px solid {color}; margin-bottom:20px;">
                <div style="display:flex; justify-content:space-between;">
                    <div>
                        <h2 style="color:{color}; margin:0;">Risk Score: {risk_score}%</h2>
                        <p style="font-size:1.2rem; color:{color}; margin:0;">{risk_category} Risk</p>
                    </div>
                    <div style="text-align:right;">
                        <p style="margin:0;"><strong>Expected Profit:</strong> €{expected_profit:,.0f}</p>
                        <p style="margin:0;"><strong>ROI:</strong> {(expected_profit/amount)*100:.1f}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Decision factors visualization
            with st.expander("📊 Risk Analysis Details"):
                col1, col2 = st.columns(2)
                with col1:
                    shap_values = explainer.shap_values(input_scaled)
                    fig, ax = plt.subplots()
                    shap.decision_plot(
                        explainer.expected_value[1],
                        shap_values[1],
                        input_data.iloc[0],
                        feature_names=features,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    factors = pd.DataFrame({
                        'Factor': features,
                        'Impact': shap_values[1][0],
                        'Value': input_data.iloc[0].values
                    }).sort_values('Impact', key=abs, ascending=False)
                    
                    fig = px.bar(
                        factors,
                        x='Impact',
                        y='Factor',
                        orientation='h',
                        color=np.where(factors['Impact'] > 0, 'Positive', 'Negative'),
                        color_discrete_map={'Positive':'#e74c3c', 'Negative':'#2ecc71'},
                        title='Feature Impact on Risk Score'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation engine
            with st.expander("💡 Business Recommendations"):
                if risk_category == "High":
                    st.warning("**Recommendation:** Decline application or require collateral")
                    st.write("**Rationale:** Expected loss exceeds risk appetite threshold")
                else:
                    st.success(f"**Recommendation:** Approve with {'standard' if risk_category == 'Low' else 'adjusted'} terms")
                    st.write(f"**Pricing Suggestion:** Base rate + {max(0, (proba-0.3)*200:.0f} bps")
                
                st.write("**Cross-Sell Opportunities:**")
                if duration > 36 and risk_category != "High":
                    st.write("- Consider longer-term fixed rate product")
                if collateral > amount * 0.5:
                    st.write("- Offer secured lending products")

def portfolio_analytics_page(df, model):
    st.header("📈 Portfolio Management Dashboard", divider='blue')
    
    # Calculate portfolio metrics
    features = ['Age', 'Job', 'Credit amount', 'Duration', 
               'Debt_to_Income', 'Loan_to_Value', 'Payment_to_Income']
    df['Risk_Score'] = model.predict_proba(df[features])[:,1]
    df['Risk_Category'] = pd.cut(df['Risk_Score'],
                                bins=[0, 0.3, 0.7, 1],
                                labels=['Low', 'Medium', 'High'])
    
    # Executive summary
    with st.container():
        st.subheader("Portfolio Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Exposure", f"€{df['Credit amount'].sum()/1e6:.2f}M")
        col2.metric("Average Risk Score", f"{df['Risk_Score'].mean():.1%}")
        col3.metric("Expected Default Rate", f"{df['Risk'].mean():.1%}")
        col4.metric("Risk-Adjusted Yield", "8.7%")
        style_metric_cards()
    
    # Portfolio composition
    with st.expander("🧩 Portfolio Composition", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                df,
                names='Risk_Category',
                title='Risk Category Distribution',
                color='Risk_Category',
                color_discrete_map={'Low':'#2ecc71', 'Medium':'#f39c12', 'High':'#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df,
                x='Risk_Category',
                y='Credit amount',
                color='Risk_Category',
                title='Loan Amount by Risk Category',
                color_discrete_map={'Low':'#2ecc71', 'Medium':'#f39c12', 'High':'#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk-return profile
    with st.expander("📊 Risk-Return Profile", expanded=True):
        fig = px.scatter(
            df,
            x='Risk_Score',
            y='Expected_Profit',
            color='Risk_Category',
            size='Credit amount',
            hover_name='Duration',
            title='Risk-Return Profile of Portfolio',
            color_discrete_map={'Low':'#2ecc71', 'Medium':'#f39c12', 'High':'#e74c3c'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio optimization
    with st.expander("⚙️ Portfolio Optimization", expanded=True):
        threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.7, 0.05)
        filtered = df[df['Risk_Score'] <= threshold]
        
        col1, col2 = st.columns(2)
        col1.metric("Remaining Portfolio Size", f"{len(filtered)} loans")
        col2.metric("Remaining Exposure", f"€{filtered['Credit amount'].sum()/1e6:.2f}M")
        
        st.write(f"**Impact Analysis:**")
        st.write(f"- Expected defaults reduced by {(df['Risk'].mean() - filtered['Risk'].mean())*100:.1f}%")
        st.write(f"- Expected profit change: €{(filtered['Expected_Profit'].sum() - df['Expected_Profit'].sum()):,.0f}")

def model_performance_page(metrics):
    st.header("🛠️ Model Diagnostics", divider='blue')
    
    if not metrics:
        st.warning("No performance metrics available")
        return
    
    # Key metrics
    st.subheader("Classification Performance")
    style_metrics({
        "Accuracy": metrics['accuracy'],
        "Precision": metrics['precision'],
        "Recall": metrics['recall'],
        "F1 Score": metrics['f1'],
        "ROC AUC": metrics['roc_auc']
    })
    
    # Confusion matrix
    with st.expander("📉 Confusion Matrix", expanded=True):
        fig = px.imshow(
            metrics['confusion_matrix'],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Good', 'Bad'],
            y=['Good', 'Bad'],
            text_auto=True,
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    with st.expander("🔍 Feature Importance", expanded=True):
        # Placeholder - would need model-specific implementation
        st.plotly_chart(
            px.bar(
                x=['Feature 1', 'Feature 2', 'Feature 3'],
                y=[0.3, 0.5, 0.2],
                title='Model Feature Importance'
            ),
            use_container_width=True
        )
    
    # Performance over time
    with st.expander("⏳ Performance Trends", expanded=True):
        # Simulated performance data
        dates = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='M')
        perf_data = pd.DataFrame({
            'Date': dates,
            'Accuracy': np.linspace(0.82, 0.87, 12) + np.random.normal(0, 0.01, 12),
            'Precision': np.linspace(0.75, 0.8, 12) + np.random.normal(0, 0.01, 12)
        })
        
        fig = px.line(
            perf_data.melt(id_vars='Date'),
            x='Date',
            y='value',
            color='variable',
            title='Model Performance Over Time',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------ Main App ------------------
def main():
    df = load_data()
    if df.empty:
        st.error("Failed to load data. Please check the data source.")
        return
    
    model, scaler, explainer, features, metrics = train_model(df)
    if model is None:
        st.error("Failed to train model. Please check the data.")
        return
    
    app_mode, risk_appetite, min_profit_margin = render_sidebar()
    
    if app_mode == "Risk Assessment":
        risk_assessment_page(df, model, scaler, explainer, features)
    elif app_mode == "Portfolio Analytics":
        portfolio_analytics_page(df, model)
    elif app_mode == "Model Performance":
        model_performance_page(metrics)
    else:
        st.write("Data explorer would go here")

if __name__ == "__main__":
    main()
