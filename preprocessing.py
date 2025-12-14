import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --- GLOBAL CONFIGURATION (Must match training exactly) ---

# 1. Define Categories for Ordinal Encoding
part_time_job_order = ["No", "Yes"]
diet_quality_order = ["Poor", "Fair", "Good"]
parental_education_level_order = ["High School", "Bachelor", "Master", "PhD"] # Added PhD to match app input
internet_quality_order = ["Poor", "Average", "Good"]
extracurricular_participation_order = ["No", "Yes"]

ordinal_categories_list = [
    part_time_job_order,
    diet_quality_order,
    parental_education_level_order,
    internet_quality_order,
    extracurricular_participation_order
]

# 2. Define Column Groups
ordinal_cols = [
    'part_time_job',
    'diet_quality',
    'parental_education_level',
    'internet_quality',
    'extracurricular_participation'
]

nominal_cols = ['gender']

# --- CUSTOM FUNCTIONS ---

def feature_engineering(df):
    """
    Custom feature engineering function.
    Must be available when loading the pipeline.
    """
    df = df.copy()

    # --- Derived Base Features ---
    if 'sleep_hours' in df.columns:
        df['sleep_deviation'] = df['sleep_hours'] - 8

    # --- Sleep Features ---
    if 'sleep_deviation' in df.columns:
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
    Selects all numeric columns from the dataframe,
    excluding those in ordinal_cols and nominal_cols.
    """
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Define what to exclude (using the globals defined above)
    exclude_cols = ordinal_cols + nominal_cols

    # Return the difference
    return [c for c in numeric_cols if c not in exclude_cols]