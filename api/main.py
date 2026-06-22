import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI(
    title="LLM Sentinel API",
    description="API de sécurité et d'analyse avec BERT",
    version="1.0.0"
)

# ==========================================
# Détection du modèle (Local vs Cloud)
# ==========================================
LOCAL_MODEL_DIR = "./models"

if os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
    MODEL_PATH = LOCAL_MODEL_DIR
    print(f"📁 Modèle local trouvé : {MODEL_PATH}")
else:
    MODEL_PATH = "prajjwal1/bert-tiny"
    print("🌐 Fallback HuggingFace")
# ==========================================
# Chargement du modèle
# ==========================================
print("⏳ Chargement du modèle BERT...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("✅ Modèle prêt !")

# ==========================================
# Schémas
# ==========================================
class AnalyzeRequest(BaseModel):
    text: str

# ==========================================
# Routes
# ==========================================
@app.get("/")
def read_root():
    return {"status": "success", "message": "Le Sentinel est opérationnel."}

@app.post("/analyze")
def analyze_text(request: AnalyzeRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Softmax → probabilités par classe
        probs = F.softmax(logits, dim=1).squeeze().tolist()

        predicted_class = torch.argmax(logits, dim=1).item()

        # Si ton modèle a 2 classes : 0 = sûr, 1 = injection
        confidence = probs[predicted_class] if isinstance(probs, list) else probs

    return {
        "input_text":      request.text,
        "predicted_class": predicted_class,
        "label":           "injection" if predicted_class == 1 else "safe",
        "confidence":      round(confidence, 4),
        "probabilities":   {
            "safe":      round(probs[0], 4) if isinstance(probs, list) else None,
            "injection": round(probs[1], 4) if isinstance(probs, list) and len(probs) > 1 else None,
        }
    }