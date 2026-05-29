import torch
from transformers import Trainer, TrainingArguments
from dataset import load_and_prep_data
from model import get_model_and_tokenizer

def tokenize_function(examples, tokenizer):
    # Cette fonction traduit tout le texte du dataset en chiffres
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def main():
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT DE LA SENTINELLE...")

    # 1. Charger le modèle et le tokenizer
    model, tokenizer = get_model_and_tokenizer()

    # 2. Charger les données
    dataset = load_and_prep_data()

    # 3. Traduire les données (Tokenization)
    print("\n⚙️ Traduction des données en Tenseurs...")
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # 4. Paramètres de l'entraînement (Comme un pro)
   # 4. Paramètres de l'entraînement (Comme un pro)
    training_args = TrainingArguments(
        output_dir="./models/llm-sentinel-tiny",
        eval_strategy="epoch",       # <--- L'orthographe exacte est ici
        learning_rate=2e-5,          
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,          
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )
    # 5. Le Professeur (Le Trainer de Hugging Face)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # 6. Lancement de l'apprentissage !
    print("\n🧠 L'IA COMMENCE À APPRENDRE... (Ça va aller très vite !)")
    trainer.train()

    # 7. Sauvegarde du modèle entraîné
    print("\n💾 Sauvegarde du modèle sur le disque dur...")
    trainer.save_model("./models/llm-sentinel-tiny_final")
    tokenizer.save_pretrained("./models/llm-sentinel-tiny_final")
    print("🎉 ENTRAÎNEMENT TERMINÉ ET SAUVEGARDÉ !")

if __name__ == "__main__":
    main()