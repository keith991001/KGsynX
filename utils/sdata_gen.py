def is_similar_to_real(new_data, real_df, numerical_cols, threshold=0.01):
    from sklearn.metrics import pairwise_distances
    import numpy as np

    real_vectors = real_df[numerical_cols].to_numpy()
    new_vector = np.array([new_data[col] for col in numerical_cols]).reshape(1, -1)
    dists = pairwise_distances(real_vectors, new_vector)
    return np.min(dists) < threshold
    
def is_valid_record(record, expected_columns):
    if not isinstance(record, dict):
        return False
    for col in expected_columns:
        if col not in record or record[col] is None:
            return False
    return True

def generate_synthetic_dataset(G, df_labeled, node_embeddings, cat_cols, num_cols, model="gpt-4o", max_samples=10):
    import openai
    import json
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import re

    synthetic_data = []
    real_df = df_labeled.copy()
    expected_columns = num_cols + cat_cols + ["target"]
    valid_patients = [n for n in G.nodes if n.startswith("patient_")][:max_samples]  # 只取前N个

    for patient_id in valid_patients:
        try:
            target_vec = node_embeddings[patient_id].reshape(1, -1)
        except KeyError:
            print(f"[!] Skipped: {patient_id} 无嵌入向量")
            continue

        similar_patients = [
            n for n in node_embeddings if n.startswith("patient_") and n != patient_id
        ]
        sim_vecs = [node_embeddings[n] for n in similar_patients]
        scores = cosine_similarity(target_vec, sim_vecs)[0]
        similar_patient_id = similar_patients[np.argmax(scores)]

        prompt = build_prompt(similar_patient_id, G)

        for _ in range(5):
            try:
                res = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一名医学数据生成专家，请严格根据结构提示生成完整患者记录。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.85,
                    top_p=0.95,
                    max_tokens=300
                )
                output = res['choices'][0]['message']['content']
                json_str = re.search(r"\{[\s\S]*\}", output)
                json_str = json_str.group(0) if json_str else None
                if not json_str:
                    print(f"[!] JSON格式失败 on {patient_id}")
                    continue

                data = json.loads(json_str)

                if is_valid_record(data, expected_columns):
                    synthetic_data.append(data)
                    break
                else:
                    print(f"[!] 字段不完整 on {patient_id}")
            except Exception as e:
                print(f"[!] Error on {patient_id}:", e)
                continue

    return pd.DataFrame(synthetic_data)
synthetic_df_sample = generate_synthetic_data(
    G=G,
    real_df=df_labeled,
    node_embeddings=node_embeddings,
    cat_cols=cat_cols,
    num_cols=num_cols,
    model="gpt-4o",      
    max_samples=10
)

synthetic_df_sample.head(10)

synthetic_df = generate_synthetic_dataset(
    G,
    df_labeled,
    node_embeddings,
    cat_cols,
    num_cols,
    model="gpt-4o",
    max_samples=len(df_labeled)  
)

