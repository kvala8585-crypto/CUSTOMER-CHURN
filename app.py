# ===============================
# CUSTOMER CHURN STREAMLIT APP
# ===============================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Customer Churn Analysis", layout="wide")

st.title("📊 Customer Churn Analysis & Prediction App")
st.markdown("**Data Analyst + Machine Learning Project (End-to-End)**")

# ===============================
# FILE PATHS (NO ERROR)
# ===============================
TRAIN_PATH = r"C:\Users\kavi vala\Desktop\CUSTOMER CHURN\churn_training.csv"
TEST_PATH  = r"C:\Users\kavi vala\Desktop\CUSTOMER CHURN\churn_testing.csv"

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

df, df_test = load_data()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("📂 Navigation")
option = st.sidebar.radio(
    "Select Section",
    [
        "Dataset Overview",
        "EDA - Churn Analysis",
        "EDA - Demographics",
        "Correlation Analysis",
        "Model Training & Evaluation",
        "Feature Importance"
    ]
)

# ===============================
# DATASET OVERVIEW
# ===============================
if option == "Dataset Overview":
    st.subheader("📄 Dataset Preview")
    st.write(df.head())

    st.subheader("📐 Dataset Shape")
    st.write(df.shape)

    st.subheader("ℹ️ Dataset Info")
    st.write(df.dtypes)

# ===============================
# EDA - CHURN
# ===============================
elif option == "EDA - Churn Analysis":
    st.subheader("📉 Churn Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x="churn", data=df, ax=ax)
    ax.set_title("Churn Distribution")
    st.pyplot(fig)

    churn_pct = df["churn"].value_counts(normalize=True) * 100
    st.write("### Churn Percentage")
    st.write(churn_pct)

# ===============================
# EDA - DEMOGRAPHICS
# ===============================
elif option == "EDA - Demographics":

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Age vs Churn")
        fig, ax = plt.subplots()
        sns.boxplot(x="churn", y="age", data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("⏳ Tenure vs Churn")
        fig, ax = plt.subplots()
        sns.boxplot(x="churn", y="tenure", data=df, ax=ax)
        st.pyplot(fig)

    st.subheader("🚻 Gender vs Churn")
    fig, ax = plt.subplots()
    sns.countplot(x="gender", hue="churn", data=df, ax=ax)
    st.pyplot(fig)

# ===============================
# CORRELATION
# ===============================
elif option == "Correlation Analysis":
    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ===============================
# MODEL TRAINING
# ===============================
elif option == "Model Training & Evaluation":

    st.subheader("🤖 Logistic Regression Model")

    X = df.drop("churn", axis=1)
    y = df["churn"]

    cat_cols = X.select_dtypes(include="object").columns
    le = LabelEncoder()

    for col in cat_cols:
        X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model Accuracy: {accuracy:.2f}")

    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

# ===============================
# FEATURE IMPORTANCE
# ===============================
elif option == "Feature Importance":

    X = df.drop("churn", axis=1)
    y = df["churn"]

    cat_cols = X.select_dtypes(include="object").columns
    le = LabelEncoder()

    for col in cat_cols:
        X[col] = le.fit_transform(X[col])

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    feature_imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.coef_[0]
    }).sort_values(by="Importance", ascending=False)

    st.subheader("⭐ Top Important Features")
    st.write(feature_imp.head(10))

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feature_imp.head(10), ax=ax)
    st.pyplot(fig)
