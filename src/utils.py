import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import matplotlib.pyplot as plt

def normalize_adjacency(adj):
   
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return torch.FloatTensor(adj_normalized.todense())

def plot_graph_results(adj, countries, preds, labels, save_path="graph_result.png"):
    G = nx.from_numpy_array(adj.todense())
    mapping = {i: name for i, name in enumerate(countries)}
    G = nx.relabel_nodes(G, mapping)

    color_map = []
    # 0: West (Blue), 1: North (Orange), 2: Rest (Green)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i in range(len(countries)):
        color_map.append(colors[preds[i]])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=color_map, 
            node_size=2500, edge_color='#bdc3c7', font_weight='bold')
    plt.title("African geopolitical chart (\colors = AI Predictions)")
    plt.savefig(save_path)
    print(f"chart saved to -> {save_path}")
    plt.close()