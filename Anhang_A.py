# Installation der benötigten Bibliotheken
!pip install datasets pandas scikit-learn imbalanced-learn

import pandas as pd
from datasets import load_dataset
from google.colab import drive

# Google Drive anbinden für dauerhafte Speicherung
drive.mount('/content/drive', force_remount=True)

# 1. Dataset von Hugging Face laden
print("Lade Datensatz...")
dataset = load_dataset("hiyata/Virus-Host-Genomes-updates-v2")

def clean_dataframe(df_raw, split_name):
    print(f"\nBereinige {split_name}-Split (Ausgangsgröße: {df_raw.shape[0]})...")
    # Fehlende Werte entfernen
    df_clean = df_raw.dropna(subset=["sequence", "host"]).copy()
    # Duplikate basierend auf der DNA-Sequenz entfernen
    df_clean = df_clean.drop_duplicates(subset=["sequence"])
    # Standard-Nukleotide filtern (A, C, G, T, N)
    df_clean["sequence"] = df_clean["sequence"].str.upper()
    df_clean = df_clean[df_clean["sequence"].str.contains(r"^[ACGTN]+$")]
    # Mindestlänge von 200 bp prüfen
    df_clean["seq_len"] = df_clean["sequence"].str.len()
    df_clean = df_clean[df_clean["seq_len"] >= 200]
    print(f"Verbleibende Sequenzen nach Bereinigung: {len(df_clean)}")
    return df_clean

# 2. Beide Splits extrahieren und bereinigen
df_train_raw = pd.DataFrame(dataset["train"])
df_test_raw = pd.DataFrame(dataset["test"])

df_train_cleaned = clean_dataframe(df_train_raw, "Train")
df_test_cleaned = clean_dataframe(df_test_raw, "Test")

# 3. Speichern der bereinigten Splits
train_save_path = '/content/drive/MyDrive/virus_host_train_cleaned.csv'
test_save_path = '/content/drive/MyDrive/virus_host_test_cleaned.csv'

df_train_cleaned.to_csv(train_save_path, index=False)
df_test_cleaned.to_csv(test_save_path, index=False)
print(f"\nTrainingsdaten gespeichert: {train_save_path}")
print(f"Testdaten gespeichert: {test_save_path}")