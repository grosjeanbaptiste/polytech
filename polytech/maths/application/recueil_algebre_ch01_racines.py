"""Recueil d'algebre, chapitre I : racines n-iemes et equations polynomiales.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 14, 15 et
18 (sections I.4.3 et I.5.3).
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.application.recueil_algebre_ch01_calculs import algebrique
from polytech.maths.domain.algebre.complexes_exacts import (
    racines_de_l_unite,
    racines_du_second_degre,
    racines_niemes_exactes,
)
from polytech.maths.domain.symboles import VARIABLE as X

IMAG: Final = sp.I
Racines = tuple[sp.Expr, ...]

RADICANDES_15: Final = {
    "a": (-10 * IMAG, 2),
    "b": (-2 + 2 * IMAG, 3),
    "c": (sp.Integer(81), 4),
    "d": (sp.Integer(-1), 5),
}
"""Lettre de l'enonce : le nombre et l'ordre de la racine."""

POLYNOMES_18: Final = {
    "a": X**7 + 3 * X**6 + 64 * X + 192,
    "d": X**4 - 6 * X**2 + 25,
    "e": X**3 + 2 * (IMAG - 1) * X**2 + (3 - 4 * IMAG) * X - 6,
}
SECONDS_DEGRES_18: Final = {"b": (-2, 5, -7), "c": (2 - 2 * IMAG, -6, 2 + 4 * IMAG)}


def exercice_14() -> dict[int, Racines]:
    """Les racines cubiques, quatriemes et sixiemes de l'unite."""
    return {n: tuple(algebrique(z) for z in racines_de_l_unite(n)) for n in (3, 4, 6)}


def exercice_15() -> dict[str, Racines]:
    """Les racines demandees, a) a c) sous forme algebrique, d) exponentielle."""
    reponses = {
        lettre: racines_niemes_exactes(z, ordre)
        for lettre, (z, ordre) in RADICANDES_15.items()
    }
    return {
        lettre: zs if lettre == "d" else tuple(algebrique(z) for z in zs)
        for lettre, zs in reponses.items()
    }


def racines_exactes(polynome: sp.Expr) -> Racines:
    """Toutes les racines, sous forme algebrique ; leve s'il en manque."""
    trouvees = sp.roots(sp.Poly(polynome, X))
    if sum(trouvees.values()) != sp.degree(polynome, X):
        msg = f"racines manquantes pour {polynome}"
        raise ArithmeticError(msg)
    return tuple(algebrique(z) for z in trouvees)


def exercice_18() -> dict[str, Racines]:
    """Les racines des cinq equations de l'exercice 18."""
    reponses = {lettre: racines_exactes(p) for lettre, p in POLYNOMES_18.items()}
    for lettre, (a, b, c) in SECONDS_DEGRES_18.items():
        reponses[lettre] = tuple(
            algebrique(z) for z in racines_du_second_degre(a, b, c)
        )
    return dict(sorted(reponses.items()))
