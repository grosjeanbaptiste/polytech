"""La reponse en frequence du biquad passe-bas de gain continu unite.

`H(s) = w_0^2 / (s^2 + (w_0 / Q) s + w_0^2)`. Tout s'ecrit en pulsation
reduite `x = w / w_0` : `|H|^2 = 1 / ((1 - x^2)^2 + x^2 / Q^2)`.

Le cours previent (chapitre 3, partie II, planche 19) que `w_0` n'est
« pas necessairement la pulsation de coupure a 3 dB ». Ce module dit ou
elle est : en `w_0` pour `Q = 1/racine(2)` seulement, au-dessus pour un
`Q` plus grand, jusqu'a `racine(1 + racine(2)) w_0`.
"""

from __future__ import annotations

import math
from typing import Final

COUPURE_SANS_AMORTISSEMENT: Final = math.sqrt(1 + math.sqrt(2))
"""`w_c / w_0` quand `Q` tend vers l'infini : 1,5538."""

_SEUIL_DE_RESONANCE: Final = 1 / math.sqrt(2)


def _exiger_une_qualite(qualite: float) -> None:
    if qualite <= 0:
        msg = f"facteur de qualite nul ou negatif : {qualite}"
        raise ValueError(msg)


def module_carre(reduite: float, qualite: float) -> float:
    """`|H(j w)|^2` a la pulsation reduite `w / w_0`."""
    _exiger_une_qualite(qualite)
    if reduite < 0:
        msg = f"pulsation negative : {reduite}"
        raise ValueError(msg)
    carre = reduite**2
    return 1 / ((1 - carre) ** 2 + carre / qualite**2)


def coupure_a_trois_decibels(qualite: float, *, cellules: int = 1) -> float:
    """`w_c / w_0` de `cellules` biquads identiques en cascade.

    Chaque cellule doit rendre `|H|^2 = 2^(-1/n)`. En `u = x^2`, c'est
    `u^2 + (Q^-2 - 2) u + 1 - 2^(1/n) = 0`, dont le produit des racines
    est negatif : une seule est positive, donc une seule coupure.
    """
    _exiger_une_qualite(qualite)
    if cellules < 1:
        msg = f"nombre de cellules inferieur a un : {cellules}"
        raise ValueError(msg)
    demi_somme = 2 - qualite**-2
    produit = 2 ** (1 / cellules) - 1
    return math.sqrt((demi_somme + math.sqrt(demi_somme**2 + 4 * produit)) / 2)


def pulsation_de_resonance(qualite: float) -> float:
    """`w_r / w_0 = racine(1 - 1 / (2 Q^2))`, le sommet du module.

    Il n'existe que pour `Q > 1/racine(2)` : en deca, le module decroit
    des l'origine.
    """
    _exiger_une_qualite(qualite)
    if qualite <= _SEUIL_DE_RESONANCE:
        msg = f"pas de resonance pour Q <= 1/racine(2) : {qualite}"
        raise ValueError(msg)
    return math.sqrt(1 - 1 / (2 * qualite**2))


def surtension(qualite: float) -> float:
    """Le module au sommet : `Q / racine(1 - 1 / (4 Q^2))`, plus grand que `Q`."""
    pulsation_de_resonance(qualite)
    return qualite / math.sqrt(1 - 1 / (4 * qualite**2))


def premiere_pulsation_d_ecart(qualite: float, ecart_en_decibels: float) -> float:
    """La plus petite `w / w_0` ou le gain depasse le continu de cet ecart.

    C'est le bord de la bande ou le gain est « plat » a cet ecart pres.
    Leve si le sommet de la reponse n'atteint pas l'ecart demande.
    """
    _exiger_une_qualite(qualite)
    if ecart_en_decibels <= 0:
        msg = f"ecart nul ou negatif : {ecart_en_decibels} dB"
        raise ValueError(msg)
    demi_somme = 2 - qualite**-2
    discriminant = demi_somme**2 - 4 * (1 - 10 ** (-ecart_en_decibels / 10))
    if demi_somme <= 0 or discriminant < 0:
        msg = f"ecart de {ecart_en_decibels} dB jamais atteint pour Q = {qualite}"
        raise ValueError(msg)
    return math.sqrt((demi_somme - math.sqrt(discriminant)) / 2)
