import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Leinwand vorbereiten
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off') # Achsen ausblenden, da es ein Schema ist

sequence = "ATGGCACTC"
k = 4

# 2. Basis-Sequenz zeichnen (Graue Kästchen)
for i, char in enumerate(sequence):
    rect = patches.Rectangle((i, 2), 1, 1, linewidth=1, edgecolor='black', facecolor='#ECF0F1')
    ax.add_patch(rect)
    ax.text(i + 0.5, 2.5, char, fontsize=16, ha='center', va='center', fontweight='bold')

# 3. Erstes k-mer Fenster (Rot)
window1 = patches.Rectangle((0, 1.95), 4, 1.1, linewidth=3, edgecolor='#E74C3C', facecolor='none', linestyle='--')
ax.add_patch(window1)
ax.annotate("", xy=(2, 1.5), xytext=(2, 1.95), arrowprops=dict(arrowstyle="->", color='#E74C3C', lw=2))
ax.text(2, 1.2, "1. K-mer: ATGG", fontsize=14, ha='center', va='center', color='#E74C3C', fontweight='bold')

# 4. Zweites k-mer Fenster (Blau) - um 1 verschoben
window2 = patches.Rectangle((1, 1.85), 4, 1.3, linewidth=3, edgecolor='#2980B9', facecolor='none', linestyle='--')
ax.add_patch(window2)
ax.annotate("", xy=(3, 0.6), xytext=(3, 1.85), arrowprops=dict(arrowstyle="->", color='#2980B9', lw=2))
ax.text(3, 0.3, "2. K-mer: TGGC", fontsize=14, ha='center', va='center', color='#2980B9', fontweight='bold')

# 5. Feinschliff und Speichern
ax.set_xlim(-1, len(sequence) + 1)
ax.set_ylim(0, 3.5)
plt.title(f"Schematische Darstellung der K-mer-Tokenisierung ($k={k}$)", fontsize=16, fontweight='bold', pad=10)
plt.tight_layout()

# In 300 dpi für die Bachelorarbeit speichern
plt.savefig('kmer_schema_300dpi.png', dpi=300, bbox_inches='tight')
print("Grafik erfolgreich als 'kmer_schema_300dpi.png' gespeichert.")