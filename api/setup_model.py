from transformers import BertTokenizer, BertForSequenceClassification
import os

print("🌐 Téléchargement du modèle depuis Hugging Face...")
# En utilisant BertTokenizer directement, on évite le bug de sentencepiece/tiktoken
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")

print("💾 Sauvegarde en cours dans le dossier ./models...")
os.makedirs("./models", exist_ok=True)
tokenizer.save_pretrained("./models")
model.save_pretrained("./models")

print("✅ Terminé ! Le modèle est maintenant 100% hors-ligne.")
