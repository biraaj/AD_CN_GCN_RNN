import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Setting random seed for consistent results
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset using GCN
class ConnectivityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        func_conn, struct_conn, time_series, label = self.data[idx]
        return (func_conn.to(DEVICE), struct_conn.to(DEVICE), time_series.to(DEVICE), label)

# GCN Layer
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        adjacency = adjacency / adjacency.sum(1, keepdim=True)
        return torch.relu(self.linear(torch.mm(adjacency, x)))

# Combined GCN and RNN Model
class GCN_RNN_Model(nn.Module):
    def __init__(self, gcn_dim, gcn_hidden_dim, rnn_hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = GCNLayer(gcn_dim, gcn_hidden_dim)
        self.gcn2 = GCNLayer(gcn_hidden_dim, gcn_dim)
        self.rnn = nn.LSTM(gcn_dim, rnn_hidden_dim, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_dim, num_classes)

    def forward(self, func_conn, struct_conn, time_series):
        def process(conn):
            adjacency = torch.eye(conn.size(0), device=DEVICE) + conn
            adjacency /= adjacency.sum(1, keepdim=True)
            x = torch.eye(conn.size(0), device=DEVICE)
            x = self.gcn2(self.gcn1(x, adjacency), adjacency)
            return x

        func_features = torch.stack([process(conn) for conn in func_conn])
        struct_features = torch.stack([process(conn) for conn in struct_conn])
        combined_features = func_features + struct_features

        temporal_features = torch.stack([
            torch.matmul(combined.T, ts).squeeze(-1)
            for combined, ts in zip(combined_features, time_series)
        ])
        _, (hidden, _) = self.rnn(temporal_features)
        return self.fc(hidden[-1])
    

# Data Loader
def load_data(dataset_root):
    data = []
    for folder in os.listdir(dataset_root):
        label = 1 if folder.startswith("AD") else 0
        func_conn = np.loadtxt(os.path.join(dataset_root, folder, "FunctionalConnectivity.txt"))
        struct_conn = np.loadtxt(os.path.join(dataset_root, folder, "StructuralConnectivity.txt"))
        time_series = np.stack([
            np.load(os.path.join(dataset_root, folder, "fmri_average_signal", file))
            for file in sorted(os.listdir(os.path.join(dataset_root, folder, "fmri_average_signal")))
            if file.endswith('.npy')
        ])

        # normalizing the values
        func_conn = torch.tensor((func_conn - func_conn.mean()) / func_conn.std(), dtype=torch.float32)
        struct_conn = torch.tensor((struct_conn - struct_conn.mean()) / struct_conn.std(), dtype=torch.float32)
        time_series = torch.tensor((time_series - time_series.mean()) / time_series.std(), dtype=torch.float32)
        data.append((func_conn, struct_conn, time_series, label))
    return data

# Evaluation with AUC-ROC Curve Plot
def evaluate_model(model, data_loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for func_conn, struct_conn, time_series, labels in data_loader:
            outputs = model(func_conn, struct_conn, time_series)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels += labels.tolist()
            all_preds += outputs.argmax(1).cpu().tolist()
            all_probs += probs.cpu().tolist()

    # Calculating metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_probs)

    print(f"Acc: {acc:.2f}, Prec: {prec:.2f}, Rec: {rec:.2f}, F1: {f1:.2f}, AUC: {auc_score:.2f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.5)
    plt.savefig("results/roc_curve_test_datase_only_spatiotemporal.png")
    #plt.show()


# Training Loop
def train_model(model, train_loader, val_loader, epochs, lr, wd):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        train_preds, train_labels, val_preds, val_labels = [], [], [], []

        for func_conn, struct_conn, time_series, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(func_conn, struct_conn, time_series)
            loss = criterion(outputs, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds += outputs.argmax(1).cpu().tolist()
            train_labels += labels.tolist()

        model.eval()
        with torch.no_grad():
            for func_conn, struct_conn, time_series, labels in val_loader:
                outputs = model(func_conn, struct_conn, time_series)
                val_loss += criterion(outputs, labels.to(DEVICE)).item()
                val_preds += outputs.argmax(1).cpu().tolist()
                val_labels += labels.tolist()

        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)

        # Average losses for the epoch
        train_loss = train_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")

    return train_losses, val_losses



## Data loading
dataset_root = "./6389Project3Data"
data = load_data(dataset_root)

train_val, test = train_test_split(data, test_size=0.2, stratify=[d[-1] for d in data], random_state=seed)  # [d[-1] for d in data] gives the labels
test_loader = DataLoader(ConnectivityDataset(test), batch_size=4, shuffle=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
avg_train_losses, avg_val_losses = [], []


## Training
for fold, (train_idx, val_idx) in enumerate(skf.split(train_val, [d[-1] for d in train_val])):
    print(f"Fold {fold + 1}")
    train_loader = DataLoader(ConnectivityDataset([train_val[i] for i in train_idx]), batch_size=4, shuffle=True)
    val_loader = DataLoader(ConnectivityDataset([train_val[i] for i in val_idx]), batch_size=4, shuffle=False)

    model = GCN_RNN_Model(gcn_dim=150, gcn_hidden_dim=128, rnn_hidden_dim=16, num_classes=2).to(DEVICE)
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=20, lr=0.0001, wd=5e-8)

    avg_train_losses.append(train_losses)
    avg_val_losses.append(val_losses)

# Mean losses across folds
avg_train_losses = np.mean(avg_train_losses, axis=0)
avg_val_losses = np.mean(avg_val_losses, axis=0)

# Loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Train Loss')
plt.plot(range(1, len(avg_val_losses) + 1), avg_val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(alpha=0.5)
plt.savefig("results/training_validation_loss_only_spatiotemporal.png")

print("\nTest Evaluation:")
evaluate_model(model, test_loader)
