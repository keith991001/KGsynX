def build_prompt(patient_id, G):
    structured_features = []
    for node in G.successors(patient_id):
        if node.startswith("target:"):
            continue
        var, val = node.split(":", 1)
        structured_features.append(f"- {var} is '{val}'")
    prompt = (
        "You are a medical data generation assistant.\n"
        "Generate a new synthetic patient record:\n"
        + "\n".join(structured_features) +
        "\nReturn as a single JSON object."
    )
    return prompt