import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Design-Einstellungen für Publikationsqualität
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# Die exakten Ergebnisse deiner Modelle
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']
rf_scores = [0.8773, 0.8061, 0.9554, 0.8744, 0.7666]
mlp_scores = [0.9016, 0.8594, 0.9326, 0.8945, 0.8050]
dnabert_scores = [0.9309, 0.8915, 0.9625, 0.9257, 0.8635]

x = np.arange(len(metrics))  # Label-Positionen
width = 0.25  # Breite der Balken

fig, ax = plt.subplots(figsize=(10, 6))

# Balken zeichnen
rects1 = ax.bar(x - width, rf_scores, width, label='Random Forest (k=4)', color='#7f8c8d')
rects2 = ax.bar(x, mlp_scores, width, label='MLP (k=4)', color='#3498db')
rects3 = ax.bar(x + width, dnabert_scores, width, label='DNABERT-2', color='#e74c3c')

# Achsenbeschriftungen und Titel
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Leistungsvergleich der Modelle zur Wirtsvorhersage', pad=20, fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold')
ax.legend(loc='lower right', frameon=True, shadow=True)

# y-Achse von 0.7 bis 1.0 limitieren, um Unterschiede besser sichtbar zu machen
ax.set_ylim(0.7, 1.0)

# Werte über den Balken anzeigen
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 Punkte vertikaler Offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, rotation=0)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

# Speichern der Grafik
output_file = "modellvergleich_metriken.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Balkendiagramm erfolgreich als '{output_file}' gespeichert!")