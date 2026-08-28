# DNA-Transformer vs. K-mer-Netzwerke: Viral Host Prediction & XAI

Dieses Repository enthält den Quellcode und die Auswertungsskripte für meine Bachelorarbeit im Studiengang **Künstliche Intelligenz und Maschinelles Lernen** an der Wilhelm Büchner Hochschule.

# Projektübersicht
Ziel dieser Arbeit ist die systematische Gegenüberstellung klassischer, frequenzbasierter K-mer-Methoden (Random Forest, Multi-Layer Perceptron) mit dem modernen DNA-Foundation-Model **DNABERT-2** zur Vorhersage des zoonotischen Potenzials und der Wirtsspezifität von Viren. Zur Auflösung der "Black-Box"-Problematik wurde zudem **Explainable AI (XAI)** mittels *Integrated Gradients* (über Captum) integriert, um genomische Hotspots zu visualisieren.

# Struktur des Repositories
- `train_dnabert2.py`: Fine-Tuning des DNABERT-2 Modells.
- `xai_analysis.py`: Extraktion genomischer Hotspots mittels Integrated Gradients (Generierung der HTML-Visualisierungen).
- `plot_metrics.py`: Skript zur Erstellung des vergleichenden Balkendiagramms der Evaluationsmetriken.
- `plot_cm_dnabert.py` / Baseline-Skripte: Generierung der Konfusionsmatrizen.
- `*.png`: Finale Ergebnis-Abbildungen (Konfusionsmatrizen und Metriken-Vergleich).

# Modellgewichte (DNABERT-2 Fine-Tuned)
Da die trainierten Modellgewichte das Git-Dateilimit überschreiten, ist das feingetunte Modell extern gespeichert:
**Download der feingetunten DNABERT-2 Gewichte:** https://huggingface.co/Brontok/dnabert2-viral-host-prediction-finetuned

# Installation & Anforderungen
Installiere die benötigten Python-Pakete via pip:
```bash
pip install -r requirements.txt
