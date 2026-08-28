import matplotlib.pyplot as plt
import seaborn as sns

# 1. Daten definieren
kategorien = ['Human', 'Non-Human']
werte = [24802, 22882]

# 2. Akademisches Styling aktivieren
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))

# 3. Balken zeichnen (dezente, professionelle Farben)
bars = sns.barplot(x=kategorien, y=werte, palette=['#2C3E50', '#E74C3C'], ax=ax, hue=kategorien, legend=False)

# 4. Achsen und Titel beschriften
ax.set_title('Verteilung der Zielvariable im bereinigten Datensatz (N = 47.684)', fontsize=14, pad=20, fontweight='bold')
ax.set_ylabel('Anzahl der Sequenzen', fontsize=12, fontweight='bold', labelpad=10)
ax.set_xlabel('Wirtskategorie (Host)', fontsize=12, fontweight='bold', labelpad=10)

# 5. Exakte Werte über die Balken schreiben
for bar in bars.patches:
    ax.annotate(f"{int(bar.get_height()):.0f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='bottom',
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=12)

# 6. Y-Achse etwas höher ansetzen, damit die Zahlen nicht abgeschnitten werden
ax.set_ylim(0, 28000)

# 7. Grafik in hoher Qualität speichern
plt.tight_layout()
plt.savefig('klassenverteilung_300dpi.png', dpi=300, bbox_inches='tight')
print("Grafik erfolgreich als 'klassenverteilung_300dpi.png' gespeichert.")