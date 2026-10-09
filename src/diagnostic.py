"""Diagnostic de l'environnement : quels composants sont réellement disponibles ?

`find_spec` teste la présence d'un module **sans l'exécuter** — un simple
`import` dans un try/except chargerait le paquet entier juste pour savoir s'il
existe.
"""
import importlib.util
import sys

import transformers

print("\n=== 🕵️ DIAGNOSTIC DE L'ENVIRONNEMENT ===")
print(f"1. Interpréteur Python actif : {sys.executable}")
print(f"2. Version de Transformers : {transformers.__version__}")

for numero, module in ((3, "sentencepiece"), (4, "tiktoken")):
    present = importlib.util.find_spec(module) is not None
    etat = "INSTALLÉ ✅" if present else "MANQUANT ❌"
    print(f"{numero}. {module.capitalize()} : {etat}")

print("========================================\n")
