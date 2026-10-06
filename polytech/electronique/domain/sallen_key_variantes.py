"""Les Sallen-Key passe-haut, passe-bande et coupe-bande (chapitre 3, partie II).

Planches 35 a 43. Le passe-haut echange les resistances et les capacites
du passe-bas ; le passe-bande ajoute une capacite `C_1` a la masse ; le
coupe-bande est un double T dont les branches de pied reviennent a la
sortie. Chaque fonction de transfert sort de l'analyse nodale.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electronique.domain.etage_de_gain import (
    ENTREE,
    transfert_autour_d_un_gain,
)
from polytech.electronique.domain.netliste import Admittance

_A, _B, _SORTIE = 2, 3, 4


def transfert_passe_haut(
    resistances: tuple[sp.Expr, sp.Expr],
    capacites: tuple[sp.Expr, sp.Expr],
    gain: sp.Expr,
    s: sp.Expr,
) -> sp.Expr:
    """Planche 35 : `C_1`, `C_2` en serie, `R_2` a la masse, `R_1` vers la sortie."""
    (r1, r2), (c1, c2) = resistances, capacites
    reseau = (
        Admittance(ENTREE, _A, s * c1),
        Admittance(_A, _B, s * c2),
        Admittance(_B, 0, 1 / sp.sympify(r2)),
        Admittance(_A, _SORTIE, 1 / sp.sympify(r1)),
    )
    return transfert_autour_d_un_gain(
        reseau, _SORTIE, commande=_B, sortie=_SORTIE, gain=gain
    )


def resistances_du_passe_haut(b0: float, b1: float, gain: float) -> tuple[float, float]:
    """`(R_1, R_2)` pour `s^2 + b_1 s + b_0`, avec `C_1 = C_2 = 1` (planche 37).

    `b_0 = 1 / (R_1 R_2)` et `2 / R_2 + (1 - mu) / R_1 = b_1` : eliminer
    `R_1` laisse `(mu - 1) b_0 R_2^2 + b_1 R_2 - 2 = 0`, dont la planche
    retient la racine positive. La formule suppose `mu > 1`.
    """
    if gain <= 1:
        msg = f"gain inferieur ou egal a un, hors de la formule : {gain}"
        raise ValueError(msg)
    if b0 <= 0 or b1 <= 0:
        msg = f"coefficient nul ou negatif : b0 = {b0}, b1 = {b1}"
        raise ValueError(msg)
    discriminant = 8 * b0 * (gain - 1) + b1**2
    seconde = (math.sqrt(discriminant) - b1) / (2 * (gain - 1) * b0)
    return 1 / (b0 * seconde), seconde


def transfert_passe_bande(
    resistances: tuple[sp.Expr, sp.Expr, sp.Expr],
    capacites: tuple[sp.Expr, sp.Expr],
    gain: sp.Expr,
    s: sp.Expr,
) -> sp.Expr:
    """Planche 38 : `R_1`, `C_1` a la masse, `C_2` serie, `R_3`, `R_2` de retour."""
    r1, r2, r3 = (sp.sympify(r) for r in resistances)
    c1, c2 = capacites
    reseau = (
        Admittance(ENTREE, _A, 1 / r1),
        Admittance(_A, 0, s * c1),
        Admittance(_A, _B, s * c2),
        Admittance(_B, 0, 1 / r3),
        Admittance(_A, _SORTIE, 1 / r2),
    )
    return transfert_autour_d_un_gain(
        reseau, _SORTIE, commande=_B, sortie=_SORTIE, gain=gain
    )


def passe_bande_a_elements_egaux(
    qualite: float, pulsation: float
) -> tuple[float, float]:
    """`(R, mu)` avec `C_1 = C_2 = 1` : `R = racine(2) / w_0`, `mu = 4 - R w_0 / Q`."""
    if qualite <= 0 or pulsation <= 0:
        msg = f"facteur de qualite ou pulsation non positif : {qualite}, {pulsation}"
        raise ValueError(msg)
    resistance = math.sqrt(2) / pulsation
    return resistance, 4 - resistance * pulsation / qualite


def transfert_coupe_bande(
    r: sp.Expr, c: sp.Expr, gain: sp.Expr, s: sp.Expr, *, alpha: sp.Expr = 1
) -> sp.Expr:
    """Planche 42 : le double T, pieds `2C` et `R/2` ramenes a la sortie.

    Le T de resistances porte `R`, `R` et `2C` ; celui de capacites `alpha C`,
    `C`, `R/2`, et `(1 - alpha) C` a la masse. Le zero est en
    `1 / (racine(alpha) R C)`, le gain continu vaut `mu`.
    """
    r, alpha = sp.sympify(r), sp.sympify(alpha)
    haut, bas, plus, sortie = 2, 3, 4, 5
    reseau = (
        Admittance(ENTREE, haut, 1 / r),
        Admittance(haut, plus, 1 / r),
        Admittance(haut, sortie, 2 * s * c),
        Admittance(ENTREE, bas, alpha * s * c),
        Admittance(bas, plus, s * c),
        Admittance(bas, sortie, 2 / r),
        Admittance(bas, 0, (1 - alpha) * s * c),
    )
    return transfert_autour_d_un_gain(
        reseau, sortie, commande=plus, sortie=sortie, gain=gain
    )
