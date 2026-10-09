# ============================================
# LLM Sentinel — Dockerfile
# ============================================
# Python 3.12 : numpy 2.4 exige >= 3.11, l'ancienne base 3.10 faisait
# echouer le build des que les dependances etaient figees.
FROM python:3.12-slim

WORKDIR /app

# torch vient de l'index CPU : l'image reste ~10x plus legere que la variante CUDA,
# et l'inference du modele tient largement sur CPU (< 50 ms).
# On l'installe en premier, puis le reste des dependances sans lui (sinon pip
# reinstallerait la variante CUDA depuis PyPI).
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && grep -v '^torch' requirements.txt > /tmp/req-app.txt \
    && pip install --no-cache-dir --timeout=300 -r /tmp/req-app.txt

# Code de l'API
COPY api/ ./api/

# Dashboard : servi par api/main.py sur /dashboard et /static.
# Il manquait ici — l'interface renvoyait 404 dans le conteneur.
COPY dashboard/ ./dashboard/

# Modele fine-tune s'il est present (sinon repli HuggingFace, voir api/main.py)
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
