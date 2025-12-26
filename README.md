# Africa-Neighbourhood-GCN
#  African Geo-GCN: Graph Neural Networks on Geopolitical Data

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red.svg)
![Status](https://img.shields.io/badge/Status-Educational-green.svg)

##  The Concept
This project explores the power of **Graph Convolutional Networks (GCNs)** applied to a custom-built dataset of African nations. 

Instead of treating countries as isolated data points (standard Machine Learning), we model the continent as a **Graph** where:
* **Nodes** are countries.
* **Edges** represent shared land borders.
* **Node Features** are economic/demographic indicators (GDP, Population).

The goal is to perform **Semi-Supervised Node Classification** to predict a country's geopolitical region based on its neighbors' influence, demonstrating how GNNs aggregate spatial information.

##  Architecture (From Scratch)
This implementation avoids high-level GNN libraries to focus on the mathematical mechanics of the **Kipf & Welling (2017)** propagation rule:

$$
H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})
$$

### Project Structure
* `src/models.py`: Custom PyTorch implementation of the Graph Convolution Layer.
* `src/data.py`: Manual construction of the African Adjacency Matrix and Feature set.
* `src/utils.py`: Spectral normalization of the adjacency matrix.

## How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
    ```
2. **Execute the Training**

    ```bash
    python main.py
    ```

## Results
The model is trained on a subset of countries and tested on "hidden" nodes (e.g., Kenya, Togo) to verify generalization.

Input Features: GDP ($B), Population (M)

Graph: 15 Nodes, Undirected.

Test Accuracy: ~100% (Converges rapidly due to strong homophily in geopolitical borders).

## Visualization
The training script automatically generates a visualization of the graph, coloring nodes by their predicted region.

---

Created with passion for AI & Africa.

**JeffreyYAJ**