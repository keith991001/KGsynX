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
    """
    Parameters:
        df (DataFrame): 输入数据，必须包含 target 列。
        features (list): 用于训练的特征名。
        cat_cols (list): 分类特征列名（用于 LabelEncoding）。

    Returns:
        model: 训练后的 RandomForestClassifier 模型。
        X_train: 编码后的训练特征。
    """
    X = encode_features(df[features], cat_cols)
    y = df["target"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_train

all_features = num_cols + cat_cols
real_model, real_X = train_model(real_df, all_features, cat_cols)
synth_model, synth_X = train_model(synthetic_df, all_features, cat_cols)

# 构建 prompt 的函数
def build_prompt(similar_patient_id, G, shap_feedback=None):
    prompt = f"Generate a synthetic patient similar to {similar_patient_id}."

    if shap_feedback:
        if isinstance(shap_feedback, dict):
            if "shap_lines" in shap_feedback:
                prompt += "\n\n" + shap_feedback["shap_lines"]
            if "diversity_instruction" in shap_feedback:
                prompt += "\n\n" + shap_feedback["diversity_instruction"]
        else:
            # 如果 shap_feedback 是字符串，也兼容拼接
            prompt += "\n\n" + str(shap_feedback)

    return prompt

def generate_synthetic_dataset(
    G,
    df_labeled,
    node_embeddings,
    cat_cols,
    num_cols,
    real_model=None,
    real_X=None,
    model="gpt-4o",
    max_samples=10,
    shap_feedback=None
):
    import openai
    import json
    import numpy as np
    import re
    from sklearn.metrics.pairwise import cosine_similarity

    # fallback：如果外部没传 shap_feedback，则自动生成一轮
    if shap_feedback is None and real_model is not None and real_X is not None:
        synth_model, synth_X = train_model(df_labeled, num_cols + cat_cols, cat_cols)
        comparison_df = compare_shap_importance(real_model, synth_model, real_X, synth_X)
        shap_feedback = generate_prompt_feedback_from_shap(comparison_df)

    synthetic_data = []
    expected_columns = num_cols + cat_cols + ["target"]
    valid_patients = [n for n in G.nodes if n.startswith("patient_")][:max_samples]

    for patient_id in valid_patients:
        try:
            target_vec = node_embeddings[patient_id].reshape(1, -1)
        except KeyError:
            print(f"[!] Skipped: {patient_id} (no embedding)")
            continue

        # 找到最相似结构的另一个患者
        similar_patients = [n for n in node_embeddings if n.startswith("patient_") and n != patient_id]
        sim_vecs = [node_embeddings[n] for n in similar_patients]
        scores = cosine_similarity(target_vec, sim_vecs)[0]
        similar_patient_id = similar_patients[np.argmax(scores)]


        for _ in range(5): 
            try:
                res = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a medical data generation assistant. Please follow the user's instructions strictly."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    top_p=0.95,
                    max_tokens=300
                )
                output = res['choices'][0]['message']['content']
                json_str = re.search(r"\{[\s\S]*\}", output)
                if not json_str:
                    continue
                data = json.loads(json_str.group(0))

                if is_valid_record(data, expected_columns):
                    synthetic_data.append(data)
                    break
            except Exception as e:
                print(f"[!] Error on {patient_id}: {e}")
                continue

    return pd.DataFrame(synthetic_data)

synthetic_df = generate_synthetic_dataset(
    G,
    df_labeled,
    node_embeddings,
    cat_cols,
    num_cols,
    real_model=real_model,
    real_X=real_X,
    model="gpt-4o",
    max_samples=len(df_labeled)
)






