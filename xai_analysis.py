import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModel
from captum.attr import LayerIntegratedGradients

# ==========================================
# 1. Konfiguration & Modellklasse
# ==========================================
model_name = "zhihan1996/DNABERT-2-117M"
model_save_path = "./DNABERT2_BestModel"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_save_path, trust_remote_code=True)

class DNABERT2ForSequenceClassification(nn.Module):
    def __init__(self, model_name, tokenizer):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.pad_token_id = tokenizer.pad_token_id
        
        original_arange = torch.arange
        def patched_arange(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return original_arange(*args, **kwargs)
        
        torch.arange = patched_arange
        try:
            self.dnabert2 = AutoModel.from_pretrained(
                model_name, config=config, trust_remote_code=True, _fast_init=False, low_cpu_mem_usage=False
            )
        finally:
            torch.arange = original_arange

        self.classifier = nn.Linear(768, 2)
        self.register_buffer('class_weights', torch.tensor([1.041954, 0.961293]))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.dnabert2(input_ids, attention_mask=attention_mask)
        embedding_mean = torch.mean(outputs[0], dim=1)
        logits = self.classifier(embedding_mean)
        return logits

# ==========================================
# 2. Modell laden
# ==========================================
print("Lade feingetuntes Modell...")
model = DNABERT2ForSequenceClassification(model_name, tokenizer)
model.load_state_dict(torch.load(f"{model_save_path}_custom_weights.pth", map_location=device))
model.to(device)
model.eval()

# ==========================================
# 3. Test-Sequenz & Captum Setup
# ==========================================
# Eine etwas längere Dummy-Sequenz, um den Effekt besser zu sehen.
# Tausche diese später gegen eine echte Sequenz aus dem Testdatensatz aus!
test_sequence = "CAAACTCAGAACTTTTTGCAATCAGCAAAATCTAACTTTAATTAATGTAAGCAGCACTTATGATGGCATCTATTATGGCACTGATGAAAAAGATAAGGCAAATCGTTACAGAATAAAAGTAAATACTACAAATCACAAAACTGTTAAAATTAAGCCACATACCAGAGAACCTCCTGCTGTACAAGAAAAACAGTTTGAATTACAAGATGCAGAAACTGATGAAAACGAATCAAAAATTCCCTCAGCTACTGTGGCAATCGTGGTGGGAGTGATTGCGGGCTTTGTAACTCTGATCATTGTCTTCATATGCTACATCTGCTGCCGCAAGCGT"

inputs = tokenizer(test_sequence, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

def custom_forward(input_ids):
    return model(input_ids=input_ids, attention_mask=attention_mask)

target_layer = model.dnabert2.embeddings.word_embeddings
lig = LayerIntegratedGradients(custom_forward, target_layer)

print("Berechne Integrated Gradients...")
attributions, delta = lig.attribute(inputs=input_ids, target=1, return_convergence_delta=True)
attributions = attributions.sum(dim=-1).squeeze(0)
attributions = attributions.cpu().detach().numpy()

# ==========================================
# 4. Premium HTML-Visualisierung generieren
# ==========================================
tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

# Dynamische Skalierung für maximalen Kontrast finden
max_attr = max(abs(attributions.max()), abs(attributions.min()))
if max_attr == 0: max_attr = 1e-9 # Division durch Null verhindern

html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; padding: 40px; color: #333; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
        h2 { margin-top: 0; color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        p { line-height: 1.6; color: #555; }
        .legend { display: flex; align-items: center; margin: 25px 0; font-size: 14px; background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef; }
        .legend-box { width: 24px; height: 24px; margin-right: 12px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .spacer { width: 30px; }
        .seq-container { display: flex; flex-wrap: wrap; gap: 6px; font-family: 'Courier New', Courier, monospace; font-size: 15px; font-weight: bold; background: #fff; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
        .token { padding: 6px 8px; border-radius: 5px; border: 1px solid rgba(0,0,0,0.05); color: #000; text-shadow: 0 0 2px rgba(255,255,255,0.8); }
    </style>
</head>
<body>
    <div class="container">
        <h2>Genomische Hotspots (DNABERT-2 XAI)</h2>
        <p>Die nachfolgende Visualisierung nutzt <strong>Integrated Gradients</strong>, um die Entscheidungsfindung des Foundation Models transparent zu machen. Die Farbintensität korreliert mit dem Einfluss des jeweiligen Token auf die Modellvorhersage.</p>
        
        <div class="legend">
            <div class="legend-box" style="background-color: rgba(220, 53, 69, 0.9);"></div> 
            <span><strong>Positiver Einfluss</strong><br>Erhöht die Wahrscheinlichkeit für "Human"</span>
            <div class="spacer"></div>
            <div class="legend-box" style="background-color: rgba(13, 110, 253, 0.9);"></div> 
            <span><strong>Negativer Einfluss</strong><br>Verringert die Wahrscheinlichkeit für "Human"</span>
        </div>
        
        <div class="seq-container">
"""

for token, score in zip(tokens, attributions):
    if token in ["[CLS]", "[SEP]", "[PAD]"]:
        continue
    
    clean_token = token.replace("##", "")
    norm_score = score / max_attr
    
    # Farbgebung: Rot für Positiv (>0), Blau für Negativ (<0)
    # Die Deckkraft (Alpha) bestimmt die Intensität
    if norm_score > 0:
        intensity = min(1.0, norm_score + 0.1) # Leichtes Anheben der Baseline für bessere Lesbarkeit
        color = f"rgba(220, 53, 69, {intensity:.3f})"
    else:
        intensity = min(1.0, abs(norm_score) + 0.1)
        color = f"rgba(13, 110, 253, {intensity:.3f})"
        
    html_content += f"<div class='token' style='background-color: {color};'>{clean_token}</div>\n"

html_content += """
        </div>
    </div>
</body>
</html>
"""

output_file = "xai_visualization.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\nFertig! Die Premium-Visualisierung wurde als '{output_file}' gespeichert.")