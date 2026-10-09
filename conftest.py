"""Préparation des tests : chemin d'import, et modèle factice pour l'API.

Deux raisons d'être :

1. Sans cela, `import api.main` échoue depuis `tests/` : pytest n'ajoute au
   chemin que le dossier du test, pas la racine du dépôt.
2. `api/main.py` charge un modèle **dès l'import**. Les tests de contrat n'ont pas
   besoin du vrai modèle fine-tuné, et ne doivent surtout pas dépendre d'un
   téléchargement réseau : on fabrique donc un classifieur minuscule et
   déterministe, et on pointe `SENTINEL_MODEL_DIR` dessus.

Le banc de détection (`tests/test_detection.py`) ignore cette variable et vise le
vrai `./models` — c'est le modèle livré qu'il doit mesurer, pas le factice.
"""

import os
import sys
import tempfile

RACINE = os.path.dirname(os.path.abspath(__file__))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from modele_factice import fabriquer  # noqa: E402  (dépend du chemin ci-dessus)

if not os.environ.get("SENTINEL_MODEL_DIR"):
    os.environ["SENTINEL_MODEL_DIR"] = fabriquer(tempfile.mkdtemp(prefix="sentinel-factice-"))
