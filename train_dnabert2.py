import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# ==========================================
# 0. Konfiguration & Pfade
# ==========================================
# Passe diesen Pfad an deinen lokalen Ordner an, in dem die CSVs liegen
data_dir = "./"  # Aktueller Ordner, oder z.B. "C:/MeinProjekt/Daten"

train_path = os.path.join(data_dir, "virus_host_train_cleaned.csv")
test_path = os.path.join(data_dir, "virus_host_test_cleaned.csv")
output_dir = "./DNABERT2_Results"
model_save_path = "./DNABERT2_BestModel"

# Prüfen, ob GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# ==========================================
# 1. Daten lokal laden
# ==========================================
print("Lade bereinigte Datensätze...")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Label Encoding (non-human = 0, human = 1)
le = LabelEncoder()
df_train['label'] = le.fit_transform(df_train['host'])
df_test['label'] = le.transform(df_test['host'])

# Umwandlung in Hugging Face Dataset
train_dataset = Dataset.from_pandas(df_train[['sequence', 'label']])
test_dataset = Dataset.from_pandas(df_test[['sequence', 'label']])

# ==========================================
# 2. DNABERT-2 Custom Model Definition
# ==========================================
print("\nLade DNABERT-2 Modell und Tokenizer...")
model_name = "zhihan1996/DNABERT-2-117M"

# Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

class DNABERT2ForSequenceClassification(nn.Module):
    def __init__(self, model_name, tokenizer):
        super().__init__()
        
        # Konfiguration laden
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.pad_token_id = tokenizer.pad_token_id
        
        # ULTIMATIVER FIX: Wir patchen PyTorch temporär!
        # Wir fangen die Funktion torch.arange ab und zwingen sie auf die CPU.
        # Das verhindert den Crash bei der Multiplikation der ALiBi-Tensoren.
        original_arange = torch.arange
        def patched_arange(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return original_arange(*args, **kwargs)
        
        torch.arange = patched_arange
        
        try:
            # Basis-Modell lokal laden
            self.dnabert2 = AutoModel.from_pretrained(
                model_name, 
                config=config, 
                trust_remote_code=True,
                _fast_init=False,
                low_cpu_mem_usage=False
            )
        finally:
            # Nach dem erfolgreichen Laden den Patch sofort wieder entfernen!
            torch.arange = original_arange
        
        # Aufgabenspezifischer Klassifikationskopf (768 -> 2 Klassen)
        self.classifier = nn.Linear(768, 2)
        
        # Class Weights als Buffer registrieren (GPU-kompatibel)
        self.register_buffer('class_weights', torch.tensor([1.041954, 0.961293]))

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.dnabert2(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        
        # Mean Pooling über die Sequenzlänge
        embedding_mean = torch.mean(hidden_states, dim=1)
        
        # Logits berechnen
        logits = self.classifier(embedding_mean)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# Modell instanziieren
model = DNABERT2ForSequenceClassification(model_name, tokenizer)

# Jetzt erst recht sicher auf die GPU schieben
model = model.to(device)

# ==========================================
# 3. Tokenisierung der Sequenzen
# ==========================================
def tokenize_function(examples):
    return tokenizer(
        examples["sequence"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

print("Tokenisiere Trainingsdaten...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
print("Tokenisiere Testdaten...")
tokenized_test = test_dataset.map(tokenize_function, batched=True)

tokenized_train = tokenized_train.remove_columns(["sequence"])
tokenized_test = tokenized_test.remove_columns(["sequence"])

# ==========================================
# 4. Definition der Evaluationsmetriken
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'f1': f1_score(labels, predictions),
        'mcc': matthews_corrcoef(labels, predictions)
    }

# ==========================================
# 5. Training (Fine-Tuning)
# ==========================================
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",       
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True if torch.cuda.is_available() else False,
    load_best_model_at_end=True,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("\nStarte lokales Fine-Tuning von DNABERT-2...")
trainer.train()

# ==========================================
# 6. Finale Evaluation & Speicherung
# ==========================================
print("\nFühre finale Evaluation auf den Testdaten aus...")
eval_results = trainer.evaluate()
print("\nErgebnisse DNABERT-2:")
for key, value in eval_results.items():
    print(f"{key}: {value}")

# Modell lokal speichern
torch.save(model.state_dict(), f"{model_save_path}_custom_weights.pth")
tokenizer.save_pretrained(model_save_path)
print(f"\nFeingetuntes Modell lokal gespeichert unter: {model_save_path}")