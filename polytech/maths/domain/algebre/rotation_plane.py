"""La rotation plane comme exponentielle : `e^A` pour `A = theta [[0, -1], [1, 0]]`.

Les puissances impaires de `A` portent `-theta` en haut a droite avec le
signe `(-1)^(k+1)` ; la serie en tire cosinus et sinus.
"""

from __future__ import annotations

import sympy as sp

_QUART_DE_TOUR = sp.Matrix([[0, -1], [1, 0]])


def generatrice(theta: sp.Expr) -> sp.Matrix:
    """`A = theta [[0, -1], [1, 0]]`."""
    return theta * _QUART_DE_TOUR


def rotation(theta: sp.Expr) -> sp.Matrix:
    """La rotation d'angle `theta`, que `e^A` doit rendre."""
    return sp.Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])


def est_la_rotation(matrice: sp.Matrix, theta: sp.Expr) -> bool:
    """Vrai si `matrice` vaut la rotation d'angle `theta`, exponentielles reecrites."""
    ecart = (matrice - rotation(theta)).applyfunc(
        lambda terme: sp.simplify(terme.rewrite(sp.exp))
    )
    return bool(ecart.is_zero_matrix)


def coin_haut_droit(matrice: sp.Matrix, k: int) -> sp.Expr:
    """Le terme en haut a droite de `A^(2k+1)`, lu sur la puissance."""
    if k < 0:
        msg = f"rang negatif : {k}"
        raise ValueError(msg)
    return (matrice ** (2 * k + 1))[0, 1]


def coin_corrige(theta: sp.Expr, k: int) -> sp.Expr:
    """`(-1)^(k+1) theta^(2k+1)`."""
    return (-1) ** (k + 1) * theta ** (2 * k + 1)


def coin_imprime(theta: sp.Expr, k: int) -> sp.Expr:
    """`(-1)^k theta^(2k+1)` : l'expression que la planche 5 imprime."""
    return (-1) ** k * theta ** (2 * k + 1)
