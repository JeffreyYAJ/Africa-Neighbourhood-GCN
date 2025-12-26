import torch
import torch.nn.functional as F
import torch.optim as optim
from src.data import load_africa_data
from src.utils import normalize_adjacency, plot_graph_results
from src.models import AfricanGCN

EPOCHS = 200
LR = 0.01
HIDDEN = 10
DROPOUT = 0.3

def main():
    # fake ETL
    adj_raw, features, labels, countries, idx_train, idx_test = load_africa_data()
    adj_norm = normalize_adjacency(adj_raw)

    model = AfricanGCN(nfeat=features.shape[1], nhid=HIDDEN, nclass=3, dropout=DROPOUT)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    
    print(f" Training for {EPOCHS} epochs\n")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_norm)
        
        loss = F.nll_loss(output[idx_train], labels[idx_train])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    model.eval()
    output = model(features, adj_norm)
    preds = output.max(1)[1]
    
    print("\n TEST SET RESULTS (Hidden Countries):")
    correct = 0
    regions = ["West", "North", "Rest"]
    
    for i in idx_test:
        pred_reg = regions[preds[i]]
        true_reg = regions[labels[i]]
        status = "0" if preds[i] == labels[i] else "X"
        if preds[i] == labels[i]: correct += 1
        print(f"Country: {countries[i]:<10} | Pred: {pred_reg:<6} | Real: {true_reg:<6} | {status}")
        
    acc = 100 * correct / len(idx_test)
    print(f"Test Accuracy: {acc:.2f}%")

    plot_graph_results(adj_raw, countries, preds, labels)

if __name__ == "__main__":
    main()