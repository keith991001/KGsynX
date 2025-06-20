import networkx as nx

def build_patient_kg(df_labeled, categorical_cols):
    G = nx.DiGraph()

    for idx, row in df_labeled.iterrows():
        patient_node = f"patient_{idx}"
        G.add_node(patient_node, type="patient")

        for col in categorical_cols:
            label_col = f"{col}_label"
            value = row[label_col]
            value_node = f"{col}:{value}"
            G.add_node(value_node, type="value", attr=col)
            G.add_edge(patient_node, value_node, relation=f"has_{col}")

        G.add_node(f"target:{int(row['target'])}", type="target")
        G.add_edge(patient_node, f"target:{int(row['target'])}", relation="has_target")

    return G


df, cat_cols, num_cols = load_uci_heart_data_typed()
df_labeled = add_categorical_labels(df)

G = build_patient_kg(df_labeled, cat_cols)

print("图中节点数：", len(G.nodes))
print("图中边数：", len(G.edges))

print("patient_0 的所有边：")
for u, v, d in G.edges("patient_0", data=True):
    print(f"{u} --[{d['relation']}]--> {v}")

import networkx as nx
import matplotlib.pyplot as plt

def visualize_subgraph_with_edge_labels(G, patients):

    sub_nodes = set()
    for p in patients:
        sub_nodes.add(p)
        sub_nodes.update(G.successors(p))

    subG = G.subgraph(sub_nodes).copy()

    # 设置颜色：患者蓝，属性值绿，诊断结果红
    color_map = []
    for node in subG.nodes:
        if node.startswith("patient_"):
            color_map.append("skyblue")
        elif node.startswith("target:"):
            color_map.append("lightcoral")
        else:
            color_map.append("lightgreen")

    edge_labels = {}
    for u, v, data in subG.edges(data=True):
        if "relation" in data:
            edge_labels[(u, v)] = data["relation"]
        else:
            edge_labels[(u, v)] = ""

    pos = nx.spring_layout(subG, seed=42, k=0.8, iterations=100)

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(subG, pos, node_color=color_map, node_size=2200)
    nx.draw_networkx_edges(subG, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(subG, pos, font_size=10)
    nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_color='black', font_size=9)

    plt.title("LOCAL KG")
    plt.axis("off")
    plt.show()


