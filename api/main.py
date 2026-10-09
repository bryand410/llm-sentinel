import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from datetime import datetime

app = FastAPI(
    title="LLM Sentinel API",
    description="API de sécurité et d'analyse avec BERT",
    version="2.0.0"
)

# ==========================================
# CORS (pour le dashboard)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Détection du modèle (Local vs Cloud)
# ==========================================
LOCAL_MODEL_DIR = os.environ.get("SENTINEL_MODEL_DIR", "./models")
FALLBACK_MODEL_ID = os.environ.get("SENTINEL_FALLBACK_MODEL", "prajjwal1/bert-tiny")

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _has_local_weights(directory: str) -> bool:
    """Un dossier de modèle n'est utilisable que s'il porte réellement les poids.

    `config.json` et le tokenizer sont versionnés dans le dépôt, les poids non
    (trop lourds). Tester la seule présence de `config.json` faisait donc croire
    à un modèle local disponible sur un clone neuf, et le chargement échouait au
    démarrage. On exige désormais un fichier de poids.
    """
    return any(os.path.exists(os.path.join(directory, f)) for f in WEIGHT_FILES)


if os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")) and _has_local_weights(LOCAL_MODEL_DIR):
    MODEL_PATH = LOCAL_MODEL_DIR
    print(f"📁 Modèle local trouvé : {MODEL_PATH}")
else:
    MODEL_PATH = FALLBACK_MODEL_ID
    print(f"🌐 Poids locaux absents — repli sur {FALLBACK_MODEL_ID}")

# ==========================================
# Chargement du modèle
# ==========================================
print("⏳ Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("✅ Modèle prêt !")

# ==========================================
# Historique en mémoire
# ==========================================
analysis_history = []

# ==========================================
# Schémas
# ==========================================
class AnalyzeRequest(BaseModel):
    text: str

# ==========================================
# Routes API
# ==========================================
@app.get("/")
def read_root():
    return {"status": "success", "message": "Le Sentinel est opérationnel."}

@app.post("/analyze")
def analyze_text(request: AnalyzeRequest):
    start = datetime.now()

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
        probs = F.softmax(logits, dim=1).squeeze().tolist()
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probs[predicted_class] if isinstance(probs, list) else probs

    elapsed_ms = (datetime.now() - start).total_seconds() * 1000

    result = {
        "input_text":      request.text,
        "predicted_class": predicted_class,
        "label":           "injection" if predicted_class == 1 else "safe",
        "confidence":      round(confidence, 4),
        "latency_ms":      round(elapsed_ms, 1),
        "probabilities":   {
            "safe":      round(probs[0], 4) if isinstance(probs, list) else None,
            "injection": round(probs[1], 4) if isinstance(probs, list) and len(probs) > 1 else None,
        }
    }

    # Garder l'historique (max 100)
    analysis_history.insert(0, {
        "timestamp": start.isoformat(),
        **result
    })
    if len(analysis_history) > 100:
        analysis_history.pop()

    return result

@app.get("/history")
def get_history():
    return {"history": analysis_history}

@app.get("/stats")
def get_stats():
    total = len(analysis_history)
    injections = sum(1 for h in analysis_history if h["label"] == "injection")
    safe = total - injections
    avg_confidence = sum(h["confidence"] for h in analysis_history) / total if total > 0 else 0
    avg_latency = sum(h["latency_ms"] for h in analysis_history) / total if total > 0 else 0
    return {
        "total_analyses": total,
        "injections_detected": injections,
        "safe_prompts": safe,
        "injection_rate": round(injections / total * 100, 1) if total > 0 else 0,
        "avg_confidence": round(avg_confidence, 4),
        "avg_latency_ms": round(avg_latency, 1),
    }

# ==========================================
# Dashboard (fichiers statiques)
# ==========================================
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard")

@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

if os.path.exists(DASHBOARD_DIR):
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")
