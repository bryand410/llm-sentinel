"""Racine du projet rendue importable pour les tests.

Sans cela, `import api.main` echoue depuis `tests/` : pytest n'ajoute au chemin
que le dossier du test, pas la racine du depot.
"""

import os
import sys

RACINE = os.path.dirname(os.path.abspath(__file__))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)
