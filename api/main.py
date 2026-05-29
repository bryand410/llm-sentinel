from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 1. Initialisation de l'API
app = FastAPI(
    title="LLM Sentinel API",
    description="API de sécurité pour bloquer les Prompt Injections",
    version="1.0.0"
)

# 2. Définition du format de la requête (Ce que l'utilisateur envoie)
class PromptRequest(BaseModel):
    text: str

# Variables globales pour stocker notre IA
model = None
tokenizer = None

# 3. Chargement de l'IA au démarrage du serveur
@app.on_event("startup")
def load_ai():
    global model, tokenizer
    print("⏳ Chargement du modèle IA Sentinel...")
    # On charge le modèle SAUVEGARDÉ sur le disque
    try:
        model_path = "./models/llm-sentinel-tiny_final"
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Modèle chargé et prêt à scanner !")
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")

# 4. La route principale : Le Scanner
@app.post("/scan")
async def scan_prompt(request: PromptRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Le texte est vide.")

    # A. L'IA traduit le texte en maths
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
    
    # B. L'IA fait sa prédiction (sans s'entraîner)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # C. Calcul des probabilités mathématiques (Softmax)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # D. Récupérer le score de l'attaque (Label 1)
    attack_score = probabilities[0][1].item()
    
    # E. Décision (Si le score est > 50%, c'est une attaque)
    is_attack = attack_score > 0.5

    return {
        "text_scanned": request.text,
        "is_attack": is_attack,
        "threat_score": round(attack_score * 100, 2), # Score en pourcentage
        "status": "🚨 BLOCKED" if is_attack else "✅ SAFE"
    }