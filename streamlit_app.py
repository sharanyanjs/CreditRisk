import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="German Credit Risk Dashboard", layout="wide")

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv")

    # Simulate Risk column
    df['Risk'] = ((df['Age'] > 30) | 
                 (df['Credit amount'] > 5000) | 
                 (df['Duration'] > 24)).astype(int)

    df = df.dropna()
    df['Risk'] = df['Risk'].map({0: 0, 1: 1})

    return df

# ------------------ Train Model ------------------
@st.cache_resource
def load_model():
    df = load_data()
    X = df[['Age', 'Job', 'Credit amount', 'Duration']]
    y = df['Risk']

    scaler = StandardScaler()
    X[['Credit amount', 'Duration']] = scaler.fit_transform(X[['Credit amount', 'Duration']])

    model = LogisticRegression()
    model.fit(X, y)

    return model, scaler

# ------------------ Load All ------------------
df = load_data()
model, scaler = load_model()

# ------------------ Title & Description ------------------
st.title("📊 German Credit Risk Prediction Dashboard")
st.markdown("""
This tool predicts the credit risk of applicants using simulated logic based on the German Credit dataset.
""")

# ------------------ Risk Prediction Form ------------------
st.header("💡 Predict Credit Risk")

with st.form("credit_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 80, 30)
        job = st.selectbox("Job Level", 
                         ["Unemployed (0)", "Unskilled (1)", "Skilled (2)", "Highly Skilled (3)"],
                         index=2)
        job = int(job.split("(")[1].replace(")", ""))
        
    with col2:
        credit_amount = st.number_input("Credit Amount (€)", min_value=100, value=5000)
        duration = st.slider("Loan Duration (months)", 6, 72, 24)
    
    submitted = st.form_submit_button("Predict Risk")
    
    if submitted:
        input_data = pd.DataFrame([{
            "Age": age,
            "Job": job,
            "Credit amount": credit_amount,
            "Duration": duration
        }])

        input_data[['Credit amount', 'Duration']] = scaler.transform(input_data[['Credit amount', 'Duration']])
        probability = model.predict_proba(input_data)[0][1]
        risk_score = round(probability * 100, 2)

        risk_color = "#ff4b4b" if risk_score > 70 else "#ffa700" if risk_score > 30 else "#00d154"
        risk_label = "High" if risk_score > 70 else "Medium" if risk_score > 30 else "Low"

        st.markdown(f"""
        <div style="background-color:{risk_color};padding:20px;border-radius:10px">
            <h2 style="color:white;text-align:center;">Risk Score: {risk_score}%</h2>
            <h3 style="color:white;text-align:center;">{risk_label} Risk</h3>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Interpretation Guide"):
            st.markdown("""
            - **0-30%**: Low Risk – Likely to repay.
            - **30-70%**: Medium Risk – Needs more review.
            - **70-100%**: High Risk – Likely to default.
            """)

# ------------------ Model Insights ------------------
st.header("📈 Model Insights")

with st.expander("Model Accuracy & Feature Importance"):
    X = df[['Age', 'Job', 'Credit amount', 'Duration']]
    y = df['Risk']
    X[['Credit amount', 'Duration']] = scaler.transform(X[['Credit amount', 'Duration']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    accuracy = model.score(X_test, y_test)

    st.success(f"✅ Model Accuracy: {accuracy*100:.2f}%")

    # Feature importance (coefficients)
    st.subheader("Feature Importance")
    features = ['Age', 'Job', 'Credit amount', 'Duration']
    importance = model.coef_[0]

    fig, ax = plt.subplots()
    ax.barh(features, importance, color='teal')
    ax.set_title("Feature Coefficients")
    st.pyplot(fig)

# ------------------ Data Exploration ------------------
st.header("🔍 Data Exploration")

st.dataframe(df.head())

col1, col2 = st.columns(2)
with col1:
    st.subheader("Risk Distribution")
    risk_counts = df['Risk'].value_counts().rename({0: 'Good', 1: 'Bad'})
    st.bar_chart(risk_counts)

with col2:
    st.subheader("Credit Amount Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['Credit amount'], bins=30, color='skyblue', edgecolor='black')
    ax.set_title('Credit Amount Histogram')
    st.pyplot(fig)
