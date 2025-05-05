from node2vec import Node2Vec

def learn_node_embeddings(G, dimensions=64):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=10, num_walks=100, workers=2, quiet=True)
    model = node2vec.fit(window=5, min_count=1)
    return {node: model.wv[node] for node in G.nodes if node in model.wv}