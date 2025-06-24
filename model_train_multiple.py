# Save this as model_train_multiple.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

# Generate synthetic data
n = 5000
df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'account_balance': np.random.randint(0, 100000, n),
    'income': np.random.randint(20000, 150000, n),
    'loan': np.random.choice(['yes', 'no'], size=n),
    'credit_card_usage': np.random.uniform(0.1, 1.0, n),
    'tenure_years': np.random.randint(1, 10, n),
    'has_mobile_app': np.random.choice(['yes', 'no'], size=n),
    'engaged': np.random.choice([0, 1], size=n, p=[0.7, 0.3])
})

X = df.drop('engaged', axis=1)
y = df['engaged']

numeric_features = ['age', 'account_balance', 'income', 'credit_card_usage', 'tenure_years']
categorical_features = ['loan', 'has_mobile_app']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

models = {
    "rf_model.pkl": RandomForestClassifier(n_estimators=100, random_state=42),
    "logreg_model.pkl": LogisticRegression(max_iter=1000),
    "xgb_model.pkl": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for filename, clf in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', clf)])
    pipeline.fit(X, y)
    joblib.dump(pipeline, filename)
    print(f"Saved {filename}")
