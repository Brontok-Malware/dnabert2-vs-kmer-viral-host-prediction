import pandas as pd
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from imblearn.under_sampling import RandomUnderSampler
from google.colab import drive

# ==========================================
# 1. Daten laden & Drive mounten
# ==========================================
# Google Drive anbinden für dauerhafte Speicherung
drive.mount('/content/drive', force_remount=True)

print("Lade bereinigte Datensätze...")
df_train = pd.read_csv('/content/drive/MyDrive/virus_host_train_cleaned.csv')
df_test = pd.read_csv('/content/drive/MyDrive/virus_host_test_cleaned.csv')

# ==========================================
# 2. K-mer Tokenisierung (k=4)
# ==========================================
def generate_kmer_features(sequences, k=4):
    # Generiere alle 256 Permutationen
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}
    
    # Feature-Matrix initialisieren
    feature_matrix = np.zeros((len(sequences), len(all_kmers)), dtype=np.float32)
    
    for idx, seq in enumerate(sequences):
        # Sliding Window Approach mit Stride = 1
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer in kmer_to_idx:
                feature_matrix[idx, kmer_to_idx[kmer]] += 1
        
        # Umwandlung in relative K-mer-Frequenzen
        total_kmers = feature_matrix[idx].sum()
        if total_kmers > 0:
            feature_matrix[idx] /= total_kmers
            
    return feature_matrix

print("Generiere K-mer Features (k=4) für Trainingsdaten...")
X_train_raw = generate_kmer_features(df_train['sequence'].values)
y_train_raw = df_train['host'].values

print("Generiere K-mer Features (k=4) für Testdaten...")
X_test = generate_kmer_features(df_test['sequence'].values)
y_test_raw = df_test['host'].values

# ==========================================
# 3. Label Encoding, Undersampling & Skalierung
# ==========================================
print("Führe Preprocessing durch...")
# Binäre Kodierung: non-human=0, human=1
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_raw) 
y_test_encoded = le.transform(y_test_raw)

# Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train_raw, y_train_encoded)

# Z-Transformation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# ==========================================
# 4. MODELL 1: Random Forest
# ==========================================
print("\n--- Training: Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train_res)

rf_preds = rf_model.predict(X_test)

print("Random Forest Ergebnisse:")
print(f"Accuracy:  {accuracy_score(y_test_encoded, rf_preds):.4f}")
print(f"Precision: {precision_score(y_test_encoded, rf_preds):.4f}")
print(f"Recall:    {recall_score(y_test_encoded, rf_preds):.4f}")
print(f"F1-Score:  {f1_score(y_test_encoded, rf_preds):.4f}")
print(f"MCC:       {matthews_corrcoef(y_test_encoded, rf_preds):.4f}")

# ==========================================
# 5. MODELL 2: Multi-Layer Perceptron (PyTorch)
# ==========================================
print("\n--- Training: Multi-Layer Perceptron (PyTorch) ---")

# Konvertierung in PyTorch Tensoren
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_res, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

# DataLoaders mit Batch Size 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Netzwerk Architektur (64 -> 32 -> 32)
class KmerMLP(nn.Module):
    def __init__(self):
        super(KmerMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = KmerMLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# Training Loop
epochs = 15
for epoch in range(epochs):
    mlp_model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = mlp_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    if (epoch + 1) % 5 == 0:
        print(f"Epoche [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Evaluation Loop
mlp_model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = mlp_model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(batch_y.numpy())

print("\nMLP Ergebnisse:")
print(f"Accuracy:  {accuracy_score(all_targets, all_preds):.4f}")
print(f"Precision: {precision_score(all_targets, all_preds):.4f}")
print(f"Recall:    {recall_score(all_targets, all_preds):.4f}")
print(f"F1-Score:  {f1_score(all_targets, all_preds):.4f}")
print(f"MCC:       {matthews_corrcoef(all_targets, all_preds):.4f}")