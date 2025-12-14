import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# --- 1. Define Column Groups ---
# Based on your dataset analysis
ordinal_cols = ['parental_education_level', 'diet_quality', 'internet_quality']
nominal_cols = ['gender', 'part_time_job', 'extracurricular_participation']

# Define mappings for Ordinal Encoder (Important to keep order)
# Adjust these lists based on your specific logical order
edu_order = ['High School', 'Bachelor', 'Master', 'PhD']
diet_order = ['Poor', 'Fair', 'Good']
net_order = ['Poor', 'Average', 'Good']

# --- 2. Custom Functions ---
def feature_engineering(df):
    """
    Recreates the feature engineering steps from the training notebook.
    """
    df = df.copy()

    # --- Derived Base Features ---
    if 'sleep_hours' in df.columns:
        df['sleep_deviation'] = df['sleep_hours'] - 8

    # --- Sleep Features ---
    df['is_severely_sleep_deprived'] = df['sleep_deviation'].apply(lambda x: 1 if x < -3.0 else 0)
    df['is_sleep_deficient'] = df['sleep_deviation'].apply(lambda x: 1 if x < -2.0 else 0)
    df['is_overslept'] = df['sleep_deviation'].apply(lambda x: 1 if x > 2.0 else 0)

    if 'social_media_hours' in df.columns and 'netflix_hours' in df.columns:
        df['total_distraction_hours'] = df['social_media_hours'] + df['netflix_hours']

    # --- Mental Health Features ---
    if 'mental_health_rating' in df.columns:
        df['mental_health_risk_score'] = df['mental_health_rating'].apply(lambda x: 1 if x < 6 else 0)
        df['mental_health_ideal_score'] = df['mental_health_rating'].apply(lambda x: 1 if x >= 8 else 0)

    # --- Exercise Features ---
    if 'exercise_frequency' in df.columns:
        df['is_sedentary'] = df['exercise_frequency'].apply(lambda x: 1 if x < 1 else 0)
        df['is_exercise_frequent'] = df['exercise_frequency'].apply(lambda x: 1 if x >= 4 else 0)

    return df

def select_numeric_cols(df):
    """
    Selects numeric columns excluding those already handled by ordinal/nominal transformers.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    exclude_cols = ordinal_cols + nominal_cols
    return [c for c in numeric_cols if c not in exclude_cols]

# --- 3. Transformers ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[edu_order, diet_order, net_order], handle_unknown='use_encoded_value', unknown_value=-1))
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, select_numeric_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('nom', nominal_transformer, nominal_cols)
    ],
    remainder='drop'
)

# Feature Engineering Transformer
feature_eng_transformer = FunctionTransformer(feature_engineering, validate=False)