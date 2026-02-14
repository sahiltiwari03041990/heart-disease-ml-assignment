import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
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
            report = classification_report(y, preds, output_dict=True)
            st.dataframe(pd.DataFrame(report).T)

        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y, preds)
            fig, ax = plt.subplots(figsize=(4, 3), dpi = 20)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted", fontsize = 5)
            ax.set_ylabel("Actual", fontsize = 5)
            st.pyplot(fig)
