from node2vec import Node2Vec
import networkx as nx

node2vec = Node2Vec(
    G,
    dimensions=64,
    walk_length=10,
    num_walks=100,
    workers=2,
    p=1, q=1,  # 可以微调探索/采样偏好
    quiet=True
)

model = node2vec.fit(window=5, min_count=1, batch_words=4)

node_embeddings = {node: model.wv[node] for node in G.nodes}

print("patient_0 embedding:", node_embeddings["patient_0"])

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = list(node_embeddings.values())
labels = list(node_embeddings.keys())

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='skyblue')

for i, label in enumerate(labels):
    if label.startswith("patient_"):
        plt.text(X_pca[i, 0], X_pca[i, 1], label, fontsize=8, color='blue')
    elif label.startswith("target:"):
        plt.text(X_pca[i, 0], X_pca[i, 1], label, fontsize=8, color='red')
    else:
        plt.text(X_pca[i, 0], X_pca[i, 1], label, fontsize=7, color='green')

plt.title("PCA Visualization of Node Embeddings")
plt.show()

