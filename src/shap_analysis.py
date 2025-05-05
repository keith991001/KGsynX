import shap
import numpy as np
import pandas as pd

def get_shap_importance(model, X):
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    return pd.Series(mean_abs_shap, index=X.columns)

def compare_shap_importance(real_model, synth_model, real_X, synth_X):
    shap_real = get_shap_importance(real_model, real_X)
    shap_synth = get_shap_importance(synth_model, synth_X)
    all_features = shap_real.index.union(shap_synth.index)
    diff = (shap_real - shap_synth).abs().reindex(all_features, fill_value=0)
    return pd.DataFrame({
        "Real": shap_real.reindex(all_features, fill_value=0),
        "Synthetic": shap_synth.reindex(all_features, fill_value=0),
        "Abs_Diff": diff
    }).sort_values(by="Abs_Diff", ascending=False)