## 1. Problem Statement

Build a Machine Learning model that predicts whether a heart failure patient will experience a death event (DEATH_EVENT) based on their clinical and demographic data. The goal is to help identify high-risk patients early so timely medical attention and monitoring can be prioritized.

This application evaluates multiple classification models to determine the outcome:
- **0 → No Heart disease**
- **1 → Having Heart disease**


## 2. Dataset Description

This dataset is a clinical dataset for heart disease prediction. It contains patient records with various medical attributes and a target variable indicating whether heart disease is present. It’s widely used in machine learning research to build classification models that predict heart disease risk.

### Dataset Source
DataSource : Kaggle

### Dataset Overview
- **Total Records:** 5000 
- **Total Columns:** 14
- **Input Features:** 13
- **Target Column:** `target`

### Attribute Details
- **age**: age of the patient (years)
- **Sex**: Gender (binary)
- **cp**: Chest pain type
- **trestbps**: Resting blood pressure (mm Hg) 
- **chol**: Serum cholesterol (mg/dl)  
- **fbs**: Fasting blood sugar > 120 mg/dl (boolean)
- **restecg**: Resting electrocardiographic results
- **thalach**: fMaximum heart rate achieved
- **exang**: Exercise-induced angina(binary) 
- **oldpeak**: ST depression induced by exercise relative to rest 
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels colored by fluoroscopy 
- **thal**: Thalassemia (blood disorder indicator)
- **target**: Diagnosis of heart disease


## 3. Models Used:

## Model Performance Comparison

| ML Model Name | Accuracy (%) | AUC | Precision | Recall | F1 Score | MCC |
|-------------|--------------|-----|----------|--------|---------|-----|
| Logistic Regression | 0.836 | 0.835 | 0.848 | 0.848 | 0.848 | 0.670 |
| Decision Tree Classifier | 0.738 | 0.739 | 0.774 | 0.727 | 0.750 | 0.476 |
| K-Nearest Neighbor Classifier | 0.836 | 0.838 | 0.871 | 0.818 | 0.844 | 0.673 |
| Naive Bayes (Gaussian) | 0.803 | 0.802 | 0.818 | 0.818 | 0.818 | 0.604 |
| Random Forest | 0.803 | 0.802 | 0.818 | 0.818 | 0.818 | 0.604 |
| XGBoost | 0.754 | 0.754 | 0.781 | 0.758 | 0.769 | 0.506 |


# Heart Disease Prediction Models - Performance Summary

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Strong baseline performance with good accuracy (0.836) and AUC (0.835). Precision (0.848) and recall (0.848) are balanced, meaning it catches most positive cases while avoiding too many false alarms. |
| Decision Tree Classifier | Moderate performance with lower accuracy (0.738) and AUC (0.739). Precision (0.774) is slightly better than recall (0.727), but single trees may overfit and not generalize well compared to ensemble methods. |
| K-Nearest Neighbor Classifier | Competitive performance with accuracy (0.836) and AUC (0.838). Precision (0.871) is strong, but recall (0.818) is slightly lower, suggesting it is cautious in predicting positives and may miss some cases. |
| Naive Bayes (Gaussian) | Solid but not top-tier performance with accuracy (0.803) and AUC (0.802). Precision and recall (both 0.818) are balanced, but the independence assumption limits its ability to capture complex relationships. |
| Random Forest | Performs similarly to Naive Bayes here, with accuracy (0.803) and AUC (0.802). Precision and recall (0.818) are balanced, but the model does not show its usual advantage, possibly due to dataset size or tuning. |
| XGBoost | Average performance compared to other models, with accuracy (0.754) and AUC (0.754). Precision (0.781) and recall (0.758) are balanced but not outstanding, indicating weaker generalization without hyperparameter optimization. |






