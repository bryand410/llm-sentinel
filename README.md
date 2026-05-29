# 🛡️ LLM Sentinel

An ultra-fast, local AI gateway designed to detect and block Prompt Injections and Jailbreaks before they hit your massive LLMs (ChatGPT, Claude, etc.).

## 🧠 Architecture
- **Model:** Fine-tuned `prajjwal1/bert-tiny` for sequence classification.
- **API:** Asynchronous FastAPI backend.
- **Latency:** < 50ms inference time on CPU.

## 🚀 Quick Start
1. `pip install -r requirements.txt`
2. Train the sentinel: `python src/train.py`
3. Launch the API: `uvicorn api.main:app --reload`