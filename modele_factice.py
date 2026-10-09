"""Fabrique un classifieur minuscule, sans réseau, pour les tests.

Utilisé par `conftest.py` (tests de contrat) et par la CI (tâche Docker). Les
tests ne doivent dépendre ni des poids fine-tunés — trop lourds pour Git — ni
d'un téléchargement HuggingFace : ils doivent tourner hors ligne, du premier
coup.

    python modele_factice.py <dossier_de_destination>
"""

import os
import sys
import tempfile

RACINE = os.path.dirname(os.path.abspath(__file__))


def fabriquer(destination: str) -> str:
    """Écrit un `BertForSequenceClassification` minuscule dans `destination`.

    Le tokenizer vient du dépôt — `models/tokenizer.json` est versionné — et le
    classifieur est créé de zéro avec des dimensions réduites. Le résultat est
    déterministe et pèse quelques centaines de kilo-octets.
    """
    from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

    os.makedirs(destination, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(RACINE, "models"))
    tokenizer.save_pretrained(destination)

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=2,
    )
    BertForSequenceClassification(config).save_pretrained(destination)
    return destination


if __name__ == "__main__":
    cible = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="sentinel-factice-")
    print(fabriquer(cible))
