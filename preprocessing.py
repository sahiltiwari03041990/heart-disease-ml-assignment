from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# define features with numerical and categorical values
numerical_cols = [ 'age', 'trestbps' ,'chol', 'thalach', 'oldpeak']

categorical_cols = [ 'sex', 'cp', 'fbs', 'restecg','exang','slope','ca','thal']

# define preprocessor to scale and encode the features and split the dataset
def get_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_cols)
        ]
    )
