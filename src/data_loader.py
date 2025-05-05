import pandas as pd

def load_uci_heart_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    df = pd.read_csv(url, names=columns)
    df = df.replace('?', pd.NA).dropna()
    for col in df.columns:
        df[col] = df[col].astype(float)
    df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=["num"])
    return df