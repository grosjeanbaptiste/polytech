"""Recueil d'algebre, chapitre VII : bases, dimensions, sommes, intersections.

Les familles sont dans `recueil_algebre_ch07_enonces`. Un sous-espace
donne par des equations -- 18, 36, 38 -- est le noyau de leur matrice.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.application import recueil_algebre_ch07_enonces as enonces
from polytech.maths.application.recueil_algebre_ch07_familles import Base, famille
from polytech.maths.domain.algebre.familles_de_vecteurs import (
    extraire_une_base,
    prolonger_en_base,
    rang_de_la_famille,
)
from polytech.maths.domain.algebre.sous_espaces import (
    base_du_noyau,
    equations,
    intersection,
    somme,
)

M = sp.Matrix
ENGENDRES: Final = ("14", "15", "25", "26", "35", "37")
EQUATIONS: Final = {
    "18A": M([[1, -2, 0]]),
    "18B": M([[1, 1, 1], [2, -1, 0]]),
    "18C": M([[1, 1, 1], [2, -1, 0], [1, -1, 2]]),
    "36W1": M([[0, 1, 1, 1]]),
    "36W2": M([[1, 2, 2, 2]]),
    "36Inter": M([[0, 1, 1, 1], [1, 2, 2, 2]]),
    "38": M([[0, 1, 3, 7]]),
}
"""Cle : la matrice des equations. Au 36, `p(0) = p(1)` s'ecrit `b + c + d = 0`
et `p(0) = 2 p(1)` s'ecrit `a + 2b + 2c + 2d = 0` ; au 38, `p(1) = p(2)`
s'ecrit `b + 3c + 7d = 0`."""
PAIRES: Final = {"16": ("14", "15"), "17": ("17S", "17T")}
A_PROLONGER: Final = ("27", "44")


def toutes() -> dict[str, Base]:
    """Une base de chaque sous-espace que le chapitre demande."""
    bases: dict[str, Base] = {k: extraire_une_base(famille(k)) for k in ENGENDRES}
    bases |= {k: base_du_noyau(m) for k, m in EQUATIONS.items()}
    bases["19"] = intersection(famille("19W1"), famille("19W2"))
    bases["29"] = somme(famille("29W1"), famille("29W2"))
    return bases


def dimensions() -> dict[str, tuple[int, int, int, int]]:
    """Exercices 16 et 17 : `dim U`, `dim V`, `dim (U + V)`, `dim (U inter V)`."""
    resultat = {}
    for cle, (premiere, seconde) in PAIRES.items():
        u, v = famille(premiere), famille(seconde)
        resultat[cle] = (
            rang_de_la_famille(u),
            rang_de_la_famille(v),
            len(somme(u, v)),
            len(intersection(u, v)),
        )
    return resultat


def exercice_22() -> tuple[tuple[sp.Expr, ...], ...]:
    """Les coefficients des equations du sous-espace engendre."""
    return equations(famille("22"))


def prolongees() -> dict[str, Base]:
    """Exercices 26, 27, 37 et 44 : une base de l'espace entier qui prolonge."""
    bases = toutes()
    resultat = {k: prolonger_en_base(famille(k)) for k in A_PROLONGER}
    resultat |= {k: prolonger_en_base(bases[k]) for k in ("26", "37")}
    return dict(sorted(resultat.items()))


def exercice_33() -> sp.Expr:
    """Le `a` pour lequel `-2 + aX + X^2` est dans le sous-espace engendre."""
    a = enonces.PARAMETRE
    (valeur,) = sp.solve(M([*famille("33"), (-2, a, 1)]).det(), a)
    return valeur
