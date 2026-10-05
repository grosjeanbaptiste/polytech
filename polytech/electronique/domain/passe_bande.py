"""Ce que la transformation passe-bas vers passe-bande fait aux pulsations.

Chapitre 3, partie I, planches 112 a 121. `s -> Omega_c (s^2 + w_0^2) /
(B s)` envoie la coupure `Omega_c` du prototype sur deux pulsations
positives `w_1'` et `w_2'` : leur ecart est la largeur `B`, et leur
moyenne **geometrique** -- non arithmetique -- est le centre `w_0`.

Chaque pole `a` du prototype en donne deux, racines de `s^2 - A B s +
w_0^2`, avec `A = a / Omega_c`. Des que `Q = w_0 / B` depasse cinq, ils
valent a peu pres `(B / 2) A +- j w_0` : le pole est reduit de `B / 2`
puis translate de `w_0` sur l'axe imaginaire.
"""

from __future__ import annotations

import cmath
import math


def _exiger_une_bande(centre: float, largeur: float) -> None:
    if largeur <= 0:
        msg = f"largeur de bande nulle ou negative : {largeur} rad/s"
        raise ValueError(msg)
    if centre <= 0:
        msg = f"pulsation centrale nulle ou negative : {centre} rad/s"
        raise ValueError(msg)


def bords_du_passe_bande(centre: float, largeur: float) -> tuple[float, float]:
    """`(w_1', w_2') = -+ B / 2 + racine(B^2 / 4 + w_0^2)`."""
    _exiger_une_bande(centre, largeur)
    racine = math.sqrt(largeur**2 / 4 + centre**2)
    return racine - largeur / 2, racine + largeur / 2


def poles_du_passe_bande(
    pole: complex, centre: float, largeur: float, *, prototype: float = 1.0
) -> tuple[complex, complex]:
    """Les deux racines de `s^2 - A B s + w_0^2`, pour `A = a / Omega_c`."""
    _exiger_une_bande(centre, largeur)
    demi = pole / prototype * largeur / 2
    ecart = cmath.sqrt(demi**2 - centre**2)
    return demi + ecart, demi - ecart


def facteur_de_qualite_de_bande(centre: float, largeur: float) -> float:
    """`Q = w_0 / B`."""
    _exiger_une_bande(centre, largeur)
    return centre / largeur
