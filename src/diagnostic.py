import sys
import transformers

print("\n=== 🕵️ DIAGNOSTIC DE L'ENVIRONNEMENT ===")
print(f"1. Interpréteur Python actif : {sys.executable}")
print(f"2. Version de Transformers : {transformers.__version__}")

try:
    import sentencepiece
    print("3. Sentencepiece : INSTALLÉ ✅")
except ImportError:
    print("3. Sentencepiece : MANQUANT ❌")

try:
    import tiktoken
    print("4. Tiktoken : INSTALLÉ ✅")
except ImportError:
    print("4. Tiktoken : MANQUANT ❌")
print("========================================\n")