"""Garde commune des comparaisons : un nan ou un inf n'est pas une mesure.

`abs(nan) > tol` est faux, donc un `nan` passait pour « egal » ; `inf - inf`
vaut `nan`. Decision : tout non-fini **leve** `ValueError` (jamais un
booleen), `inf` compris, meme face a lui-meme -- l'infini d'un resultat
est un defaut a voir, pas une egalite a constater.
"""

from __future__ import annotations

import cmath


def exiger_fini(**valeurs: complex) -> None:
    """Leve `ValueError` nommant le premier argument nan ou infini."""
    for nom, valeur in valeurs.items():
        if not cmath.isfinite(valeur):
            msg = (
                f"valeur non finie pour {nom} : {valeur} (nan/inf ne se comparent pas)"
            )
            raise ValueError(msg)
