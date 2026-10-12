"""Recueil d'algebre, chapitre I : conditions lues dans le plan de Gauss.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 2, 12 et
13. On pose `z = x + iy` ; chaque fonction rend l'expression en `x` et `y`
dont l'annulation -- ou le signe -- decrit l'ensemble.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.algebre.complexes_exacts import (
    ORDONNEE as Y,
)
from polytech.maths.domain.algebre.complexes_exacts import (
    lieu_dans_le_plan_de_gauss,
)
from polytech.maths.domain.symboles import VARIABLE as X


def exercice_2() -> tuple[sp.Expr, sp.Expr]:
    """Parties reelle et imaginaire de `(x + iy)^4`."""
    reelle, imaginaire = sp.expand((X + sp.I * Y) ** 4).as_real_imag()
    return sp.expand(reelle), sp.expand(imaginaire)


def exercice_12() -> dict[str, sp.Expr]:
    """Les quatre ensembles : chacun est le lieu ou l'expression s'annule.

    `D` demande `z / conj(z) = i` ; on l'ecrit `z - i conj(z) = 0`, valable
    hors de l'origine, et l'on en garde la partie reelle `x - y`.
    """
    return {
        "A": lieu_dans_le_plan_de_gauss(lambda z: sp.re(z) + sp.im(z) - 3),
        "B": lieu_dans_le_plan_de_gauss(lambda z: sp.re(z**2)),
        "C": lieu_dans_le_plan_de_gauss(lambda z: sp.re(z + sp.conjugate(z)) / 2),
        "D": lieu_dans_le_plan_de_gauss(lambda z: sp.re(z - sp.I * sp.conjugate(z))),
    }


def exercice_13() -> dict[str, sp.Expr]:
    """`A` : negatif ou nul dans le disque ; `B` : negatif dans le demi-plan."""
    return {
        "A": lieu_dans_le_plan_de_gauss(lambda z: sp.Abs(z) ** 2 - 9),
        "B": lieu_dans_le_plan_de_gauss(
            lambda z: sp.Abs(z - 1) ** 2 - sp.Abs(z - 2) ** 2
        ),
    }
