import pandas as pd
from modules.train_model import train_model
from modules.shap_tools import compare_shap_importance, generate_prompt_feedback_from_shap

def run_refinement_loop(real_df, generate_fn, G, node_embeddings, cat_cols, num_cols,
                        max_rounds=5, model_name="gpt-4o", shap_diff_threshold=0.03):
    all_features = num_cols + cat_cols
    real_model, real_X = train_model(real_df, all_features, cat_cols)
    shap_feedback = None
    best_synthetic_df = None
    best_shap_diff = float("inf")

    for round_idx in range(1, max_rounds + 1):
        print(f"=== Round {round_idx} ===")
        synthetic_df = generate_fn(G=G, df_labeled=real_df, node_embeddings=node_embeddings,
                                   cat_cols=cat_cols, num_cols=num_cols,
                                   real_model=real_model, real_X=real_X,
                                   model=model_name, max_samples=len(real_df),
                                   shap_feedback=shap_feedback)
        synth_model, synth_X = train_model(synthetic_df, all_features, cat_cols)
        shap_df = compare_shap_importance(real_model, synth_model, real_X, synth_X)
        shap_diff_score = shap_df["Abs_Diff"].sum()
        print(f"SHAP diff: {shap_diff_score:.4f}")

        if shap_diff_score < best_shap_diff:
            best_synthetic_df = synthetic_df.copy()
            best_shap_diff = shap_diff_score

        if shap_diff_score < shap_diff_threshold:
            break

        shap_feedback = generate_prompt_feedback_from_shap(shap_df)

    return best_synthetic_df
