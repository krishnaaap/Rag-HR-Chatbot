import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# -------------------- 1. Load Dataset -------------------- #
df = pd.read_csv("multi_intent_dataset.csv")

# Replace NaN in 'greeting' column with 0
df["greeting"] = df["greeting"].fillna(0)

# Ensure labels are integers
df[["policy related", "leave intension", "greeting"]] = df[["policy related", "leave intension", "greeting"]].astype(int)

# Show label combination counts for balance check
print("\nLabel combination counts:")
print(df.groupby(["policy related", "leave intension", "greeting"]).size())

# -------------------- 2. Prepare Data -------------------- #
texts = df["text"].tolist()
labels = df[["policy related", "leave intension", "greeting"]].values.tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# -------------------- 3. Tokenizer -------------------- #
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class MultiIntentDataset(Dataset):
    """Custom dataset for multi-label intent classification."""
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

train_dataset = MultiIntentDataset(train_texts, train_labels, tokenizer)
val_dataset = MultiIntentDataset(val_texts, val_labels, tokenizer)

# -------------------- 4. Model -------------------- #
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    problem_type="multi_label_classification"
)

# -------------------- 5. Training Arguments -------------------- #
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="epoch"
)

# -------------------- 6. Trainer -------------------- #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# -------------------- 7. Train -------------------- #
trainer.train()

# -------------------- 8. Evaluate -------------------- #
eval_results = trainer.evaluate()
print("\n📊 Evaluation Results:", eval_results)

# -------------------- 9. Save Model -------------------- #
model.save_pretrained("./multi_intent_bert")
tokenizer.save_pretrained("./multi_intent_bert")

print("✅ Model training complete. Saved in ./multi_intent_bert")
