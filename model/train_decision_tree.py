import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from preprocessing import get_preprocessor

df = pd.read_csv("data/heart.csv")
# Remove duplicates based only on feature columns
feature_cols = df.columns.drop("target")
df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline(
    steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]
)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "model/decision_tree.pkl")
print("✅ Decison Tree classifier saved successfully")
