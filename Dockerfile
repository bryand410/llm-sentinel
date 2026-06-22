# 1. On part d'un ordinateur Linux ultra-léger avec Python déjà installé
FROM python:3.10-slim

# 2. On définit le dossier de travail à l'intérieur de la boîte
WORKDIR /app

# 3. On copie seulement le fichier des dépendances en premier
COPY requirements.txt .

# 4. On installe la version CPU de PyTorch en premier (Pour économiser 2.5 Go !)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 5. On copie tout le reste du code
COPY . .

# 6. On expose le port 8000 pour que le monde extérieur puisse parler à l'API
EXPOSE 8000

# 7. La commande pour allumer le serveur quand la boîte démarre
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]