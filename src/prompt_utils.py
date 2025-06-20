#用embedding寻找结构最相似的病人

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_patients(target_patient, node_embeddings, top_k=3):
    patient_nodes = [n for n in node_embeddings if n.startswith("patient_") and n != target_patient]
    target_vec = node_embeddings[target_patient].reshape(1, -1)
    patient_vecs = [node_embeddings[p] for p in patient_nodes]

    similarities = cosine_similarity(target_vec, patient_vecs)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [patient_nodes[i] for i in top_indices]

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
        "Only include the following fields (no extra fields):\n"
        f"{required_fields}\n\n"
        "Structural features:\n"
        + "\n".join(structured_features)
        + "\n\nReturn a single JSON object only.\n"
    )

    # 添加 SHAP 反馈
    if shap_feedback:
        prompt += "\nAdditional generation guidelines based on model interpretability:\n"
        prompt += shap_feedback

    return prompt


