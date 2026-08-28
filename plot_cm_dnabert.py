import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model_name = "zhihan1996/DNABERT-2-117M"
model_save_path = "./DNABERT2_BestModel"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Klassendefinition (mit Patch)
# ==========================================
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

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.dnabert2(input_ids, attention_mask=attention_mask)
        embedding_mean = torch.mean(outputs[0], dim=1)
        logits = self.classifier(embedding_mean)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# ==========================================
# 2. Daten laden & Tokenisieren
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(model_save_path, trust_remote_code=True)

df_test = pd.read_csv("virus_host_test_cleaned.csv")
df_test['label'] = df_test['host'].apply(lambda x: 1 if x == "human" else 0)
test_dataset = Dataset.from_pandas(df_test[['sequence', 'label']])

def tokenize_function(examples):
    return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=512)

print("Tokenisiere Testdaten...")
tokenized_test = test_dataset.map(tokenize_function, batched=True).remove_columns(["sequence"])

# ==========================================
# 3. Modell laden & Vorhersagen generieren
# ==========================================
print("Lade Modell für Evaluation...")
model = DNABERT2ForSequenceClassification(model_name, tokenizer)
model.load_state_dict(torch.load(f"{model_save_path}_custom_weights.pth", map_location=device))
model.to(device)
model.eval()

print("Generiere Vorhersagen (das dauert kurz)...")
trainer = Trainer(model=model)
predictions = trainer.predict(tokenized_test)

logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
# FIX: Wir drehen die Vorhersagen um (1 - x), da das Modell "human" als 0 gelernt hat, 
# wir es im Plot aber als 1 darstellen wollen.
preds = 1 - np.argmax(logits, axis=-1)

# ==========================================
# 4. Matrix plotten
# ==========================================
cm = confusion_matrix(predictions.label_ids, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Human", "Human"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Konfusionsmatrix: DNABERT-2")
plt.savefig("cm_dnabert.png", dpi=300, bbox_inches='tight')
print("DNABERT-2 Matrix erfolgreich als 'cm_dnabert.png' gespeichert!")