"""Deux droites paralleles de charges opposees : le probleme plan de base.

Le syllabus d'electricite generale (exemple 3.8.2) : la droite 1, de charge
lineique `lambda_1`, perce le plan en `x = -a` ; la droite 2, de charge
`-lambda_1`, en `x = +a`. La somme des deux potentiels logarithmiques vaut

    V = (lambda_1 / 4 pi eps_0) ln{[(x - a)^2 + y^2] / [(x + a)^2 + y^2]}   (3.8.9)

Les equipotentielles sont les cercles `rho_2 / rho_1 = p`, avec
`p = exp(2 pi eps_0 V / lambda_1)` (3.8.6), de centre
`x_0 = -a (p^2 + 1) / (p^2 - 1)` et de rayon `|2 a p / (p^2 - 1)|` (3.8.13).
Les lignes de champ sont les cercles qui passent par les deux droites
(3.8.15). Les deux familles sont orthogonales.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.symboles import ABSCISSE, ORDONNEE


def potentiel_de_deux_droites(lineique: object, demi_ecart: object) -> sp.Expr:
    """Le potentiel (3.8.9), nul sur le plan mediateur `x = 0`."""
    a = sp.sympify(demi_ecart)
    vers_la_seconde = (ABSCISSE - a) ** 2 + ORDONNEE**2
    vers_la_premiere = (ABSCISSE + a) ** 2 + ORDONNEE**2
    facteur = sp.sympify(lineique) / (4 * sp.pi * PERMITTIVITE)
    return facteur * sp.log(vers_la_seconde / vers_la_premiere)


def parametre_de_l_equipotentielle(potentiel: object, lineique: object) -> sp.Expr:
    """`p = exp(2 pi eps_0 V / lambda_1)`, soit `rho_2 / rho_1` a ce potentiel."""
    return sp.exp(
        2 * sp.pi * PERMITTIVITE * sp.sympify(potentiel) / sp.sympify(lineique)
    )


def cercle_equipotentiel(parametre: Any, demi_ecart: Any) -> tuple[Any, Any]:  # noqa: ANN401 -- exacts, reels ou symboliques
    """`(x_0, R)` : l'abscisse du centre et le rayon de l'equipotentielle `p`."""
    p, a = sp.sympify(parametre), sp.sympify(demi_ecart)
    if p == 1:
        msg = "p = 1 designe le plan mediateur, qui n'est pas un cercle"
        raise ValueError(msg)
    return (-a * (p**2 + 1) / (p**2 - 1), sp.Abs(2 * a * p / (p**2 - 1)))


def cercle_de_champ(constante: Any, demi_ecart: Any) -> tuple[tuple[Any, Any], Any]:  # noqa: ANN401 -- exacts, reels ou symboliques
    """Le centre `(0, K a / 2)` et le rayon `a sqrt(1 + K^2 / 4)` du cercle `K`."""
    k, a = sp.sympify(constante), sp.sympify(demi_ecart)
    return ((sp.Integer(0), k * a / 2), a * sp.sqrt(1 + k**2 / 4))
