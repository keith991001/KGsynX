import pandas as pd

def load_uci_heart_data_typed():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal", "num"
    ]
    df = pd.read_csv(url, names=columns)
    df = df.replace('?', pd.NA).dropna()

    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    for col in categorical_cols + numerical_cols + ['num']:
        df[col] = df[col].astype(float)

    df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=["num"])

    return df, categorical_cols, numerical_cols

def add_categorical_labels(df):
    df = df.copy()
    mappings = {
        'sex': {0.0: 'female', 1.0: 'male'},
        'cp': {
            1.0: 'typical_angina', 2.0: 'atypical_angina',
            3.0: 'non_anginal_pain', 4.0: 'asymptomatic'
        },
        'fbs': {0.0: 'false', 1.0: 'true'},
        'restecg': {
            0.0: 'normal', 1.0: 'ST-T_abnormality', 2.0: 'left_ventricular_hypertrophy'
        },
        'exang': {0.0: 'no', 1.0: 'yes'},
        'slope': {1.0: 'upsloping', 2.0: 'flat', 3.0: 'downsloping'},
        'thal': {3.0: 'normal', 6.0: 'fixed_defect', 7.0: 'reversible_defect'},
        'ca': {0.0: '0_vessels', 1.0: '1_vessel', 2.0: '2_vessels', 3.0: '3_vessels'}
    }

    for col, mapping in mappings.items():
        df[f"{col}_label"] = df[col].map(mapping)

    return df
