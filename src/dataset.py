from datasets import load_dataset

def load_and_prep_data():
    print("📥 Téléchargement des données de sécurité depuis Hugging Face...")
    
    # Téléchargement du dataset (0 = texte normal, 1 = attaque/prompt injection)
    dataset = load_dataset("deepset/prompt-injections")
    
    print("\n✅ Données téléchargées avec succès !")
    print(f"📊 Nombre d'exemples d'entraînement : {len(dataset['train'])}")
    print(f"📊 Nombre d'exemples de test : {len(dataset['test'])}")
    
    # Chercher un exemple normal
    print("\n🟢 EXEMPLE NORMAL (Label 0) :")
    safe_example = next(item for item in dataset['train'] if item['label'] == 0)
    print(f"Texte : '{safe_example['text']}'")
    
    # Chercher un exemple de piratage
    print("\n🔴 EXEMPLE D'ATTAQUE / JAILBREAK (Label 1) :")
    hacked_example = next(item for item in dataset['train'] if item['label'] == 1)
    print(f"Texte : '{hacked_example['text']}'")
    
    return dataset

# Ce bloc permet d'exécuter la fonction si on lance ce fichier directement
if __name__ == "__main__":
    data = load_and_prep_data()