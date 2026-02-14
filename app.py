import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,   confusion_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("🫀 Heart Disease Prediction – ML Models")

st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["logistic_regression","decision_tree","knn","naive_bayes","random_forest","xgboost"]
)

file_path = "data/heart_test_with_target.csv"

uploaded_file = st.file_uploader("Upload CSV test data", type=["csv"])

with open(file_path, "rb") as f:
        csv_data = f.read()

st.download_button(
        label="⬇️ Download Test CSV",
        data=csv_data,
        file_name="test_dataset_without_target.csv",
        mime="text/csv"
    )


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'target' in data.columns:
        X = data.drop('target', axis=1)
        y = data['target']
    else:
        X = data
        y = None

    model = joblib.load(f"model/{model_name}.pkl")
    preds = model.predict(X)


# upload test.csv file and download options
    col1, col2 = st.columns([3,3])

    if y is not None:
        with col1:
            st.subheader("Evaluation Metrics")
            accuracy = accuracy_score(y, preds)
            precision = precision_score(y, preds)
            recall = recall_score(y, preds)
            f1 = f1_score(y, preds)
            mcc = matthews_corrcoef(y, preds)
            auc = roc_auc_score(y, preds)
            
            c1,c2 = st.columns(2)
            c1.metric("Accuracy", f"{accuracy:.3f}")
            c2.metric("Precision", f"{precision:.3f}")

            c3,c4 = st.columns(2)
            c3.metric("Recall", f"{recall:.3f}")
            c4.metric("F1 Score", f"{f1:.3f}")
            
            c5, c6 = st.columns(2)
            c5.metric("MCC Score", f"{mcc:.3f}")
            c6.metric("AUC Score", f"{auc:.3f}" if auc is not None else "N/A")

        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y, preds)
            fig, ax = plt.subplots(figsize=(4, 3), dpi = 20)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted", fontsize = 5)
            ax.set_ylabel("Actual", fontsize = 5)
            st.pyplot(fig)
