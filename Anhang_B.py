import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# 1. Bereinigten Trainingsdatensatz laden
df_train = pd.read_csv('/content/drive/MyDrive/virus_host_train_cleaned.csv')

print("Verteilung der Zielvariable 'host' im Trainings-Split:")
print(df_train["host"].value_counts(normalize=True))
print(df_train["host"].value_counts())

# --- Strategie 1: Undersampling für klassische Modelle ---
min_class_count = df_train["host"].value_counts().min()

df_train_balanced = df_train.groupby("host", group_keys=False).apply(
    lambda x: x.sample(n=min_class_count, random_state=42)
)

print("\nVerteilung im Trainingsdatensatz nach Undersampling:")
print(df_train_balanced["host"].value_counts())

# --- Strategie 2: Klassengewichtung für Deep-Learning-Modelle ---
classes = np.unique(df_train["host"])
class_weights = compute_class_weight(
    class_weight="balanced", 
    classes=classes, 
    y=df_train["host"]
)
class_weight_dict = dict(zip(classes, class_weights))
print("\nBerechnete Class Weights (Trainingsdaten):", class_weight_dict)