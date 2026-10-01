"""Les deux gardes que tout objet-valeur du contexte reutilise.

Un `nan` ou un `inf` qui entre dans une grandeur se propage ensuite dans
toutes les macros du corrige sans rien casser : le document compose, et
affiche `nan`. La garde est donc au plus tot, a la construction.
"""

from __future__ import annotations

import math


def exiger_fini(valeur: float, grandeur: str) -> None:
    """Leve si la valeur n'est ni finie ni representable."""
    if not math.isfinite(valeur):
        msg = f"{grandeur} : valeur non finie ({valeur})"
        raise ValueError(msg)


def exiger_strictement_positif(valeur: float, grandeur: str) -> None:
    """Leve si la valeur est nulle ou negative -- elle divise quelque part."""
    exiger_fini(valeur, grandeur)
    if valeur <= 0:
        msg = f"{grandeur} nulle ou negative interdite : {valeur}"
        raise ValueError(msg)
