import pandas as pd
import numpy as np
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def encode_features(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def train_model(df, features, cat_cols):
    X = encode_features(df[features], cat_cols)
    y = df["target"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train

def get_shap_importance(model, X):
    import shap
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    return pd.Series(mean_abs_shap, index=X.columns)

def compare_shap_importance(real_model, synth_model, real_X, synth_X):
    shap_real = get_shap_importance(real_model, real_X)
    shap_synth = get_shap_importance(synth_model, synth_X)
    all_features = shap_real.index.union(shap_synth.index)
    shap_real = shap_real.reindex(all_features, fill_value=0)
    shap_synth = shap_synth.reindex(all_features, fill_value=0)
    diff = (shap_real - shap_synth).abs()
    return pd.DataFrame({
        "Real": shap_real,
        "Synthetic": shap_synth,
        "Abs_Diff": diff
    }).sort_values(by="Abs_Diff", ascending=False)

def generate_prompt_feedback_from_shap(comparison_df, threshold=0.03):
    prompt_lines = []
    for feature, row in comparison_df.iterrows():
        diff = row["Abs_Diff"]
        if diff >= threshold:
            if row["Real"] > row["Synthetic"]:
                msg = f"The feature '{feature}' is underrepresented in synthetic data."
            else:
                msg = f"The feature '{feature}' is overrepresented in synthetic data."
            prompt_lines.append(msg)
    return "\n".join(f"- {line}" for line in prompt_lines)

def build_prompt(patient_id, G, shap_feedback=None):
    required_fields = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    structured_features = []
    for node in G.successors(patient_id):
        if node.startswith("target:"):
            continue
        var, val = node.split(":", 1)
        structured_features.append(f"- {var} is '{val}'")
    prompt = (
        "You are a medical data generation assistant.\n"
        "Generate a new synthetic patient record based on the following structural features.\n"
        f"Only include these fields: {required_fields}\n\n"
        "Structural features:\n"
        + "\n".join(structured_features)
        + "\n\nReturn a single JSON object only.\n"
    )
    if shap_feedback:
        prompt += "\nAdditional feedback:\n" + shap_feedback
    return prompt

def is_valid_record(record, expected_columns):
    if not isinstance(record, dict):
        return False
    for col in expected_columns:
        if col not in record or record[col] is None:
            return False
    return True

def generate_synthetic_dataset(G, df_labeled, node_embeddings, cat_cols, num_cols, generate_api_fn, feedback=None, max_samples=10):
    synthetic_data = []
    expected_columns = num_cols + cat_cols + ["target"]
    valid_patients = [n for n in G.nodes if n.startswith("patient_")][:max_samples]
    for patient_id in valid_patients:
        try:
            vec = node_embeddings[patient_id].reshape(1, -1)
        except:
            continue
        similar_patients = [n for n in node_embeddings if n.startswith("patient_") and n != patient_id]
        sim_vecs = [node_embeddings[n] for n in similar_patients]
        scores = cosine_similarity(vec, sim_vecs)[0]
        similar_patient = similar_patients[np.argmax(scores)]
        prompt = build_prompt(similar_patient, G, feedback)
        for _ in range(3):
            output = generate_api_fn(prompt)
            try:
                json_str = re.search(r"\{[\s\S]*\}", output).group(0)
                data = json.loads(json_str)
                if is_valid_record(data, expected_columns):
                    synthetic_data.append(data)
                    break
            except:
                continue
    return pd.DataFrame(synthetic_data)

def run_refinement_loop(G, real_df, df_labeled, node_embeddings, cat_cols, num_cols, generate_api_fn, max_rounds=3):
    features = num_cols + cat_cols
    real_model, real_X = train_model(real_df, features, cat_cols)
    shap_feedback = None
    best_df = None
    best_score = float("inf")
    for i in range(max_rounds):
        synthetic_df = generate_synthetic_dataset(G, df_labeled, node_embeddings, cat_cols, num_cols, generate_api_fn, shap_feedback, max_samples=len(real_df))
        synth_model, synth_X = train_model(synthetic_df, features, cat_cols)
        comp_df = compare_shap_importance(real_model, synth_model, real_X, synth_X)
        score = comp_df["Abs_Diff"].sum()
        if score < best_score:
            best_df = synthetic_df.copy()
            best_score = score
        shap_feedback = generate_prompt_feedback_from_shap(comp_df)
    return best_df