import networkx as nx

def build_kg(df, categorical_cols):
    G = nx.DiGraph()
    for idx, row in df.iterrows():
        patient_node = f"patient_{idx}"
        G.add_node(patient_node, type="patient")
        for col in categorical_cols:
            value_node = f"{col}:{row[col]}"
            G.add_node(value_node, type="value")
            G.add_edge(patient_node, value_node, relation=f"has_{col}")
        G.add_node(f"target:{int(row['target'])}", type="target")
        G.add_edge(patient_node, f"target:{int(row['target'])}", relation="has_target")
    return G