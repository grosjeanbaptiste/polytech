"""Mathematique, exercices 1, section 2 : matrices, determinants, systemes.

L'inversion d'une matrice a parametre par sa comatrice, trois systemes de
Cramer, et deux systemes homogenes a parametres : ceux-la n'ont de
solution non nulle que si leur determinant s'annule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.domain.algebre.determinants import Determinant
from polytech.maths.domain.algebre.systemes import SystemeLineaire

if TYPE_CHECKING:
    from collections.abc import Mapping

PARAMETRE: Final = sp.Symbol("a", real=True)
SECONDS_MEMBRES: Final = sp.symbols("a b c", real=True)
"""Les trois parametres de l'exercice 2.2.3, puis de la section 2.3."""
_A, _B, _C = SECONDS_MEMBRES
SYSTEMES: Final = (
    (sp.Matrix([[1, 1, -1], [1, -1, 0], [1, 4, 1]]), sp.Matrix([0, 0, 0])),
    (sp.Matrix([[1, 1, 2], [1, -1, -1], [1, 0, 1]]), sp.Matrix([5, 1, 3])),
    (sp.Matrix([[3, -1, 2], [-1, 2, -3], [1, 2, 1]]), sp.Matrix([_A, _B, _C])),
)
PARAMETRIQUES: Final = {
    1: sp.Matrix([[1, 1, 1], [_B + _C, _C + _A, _A + _B], [_B * _C, _A * _C, _A * _B]]),
    2: sp.Matrix([[1, -_A, -_A], [-_B, 1, -_B], [-_C, -_C, 1]]),
}
"""Les matrices des deux systemes homogenes de la section 2.3."""


def matrice() -> sp.Matrix:
    """La matrice `A` de l'exercice 2.1."""
    a = PARAMETRE
    return sp.Matrix([[1, 0, a], [0, a, 1], [a, 1, 0]])


def determinant() -> sp.Expr:
    """`det A`, developpe selon la premiere ligne."""
    return Determinant(matrice()).par_cofacteurs(0)


def developpement() -> tuple[sp.Expr, ...]:
    """Les trois termes du developpement, avant de les sommer."""
    return Determinant(matrice()).termes_du_developpement(0)


def valeurs_singulieres() -> tuple[sp.Expr, ...]:
    """Les valeurs de `a` ou `A` n'est pas inversible.

    Le parametre est declare reel : la resolution ne rend donc que `-1`,
    et laisse de cote les deux racines complexes de `1 + a^3`.
    """
    return Determinant(matrice()).valeurs_singulieres(PARAMETRE)


def comatrice() -> sp.Matrix:
    """La matrice des cofacteurs de `A`."""
    return Determinant(matrice()).comatrice()


def inverse() -> sp.Matrix:
    """`A^-1 = tA~ / det A`, pour `a` different de -1."""
    return sp.simplify(Determinant(matrice()).inverse())


def solutions_des_systemes() -> tuple[tuple[sp.Expr, ...], ...]:
    """La solution unique de chacun des trois systemes de l'exercice 2.2."""
    solutions = []
    for coefficients, second_membre in SYSTEMES:
        systeme = SystemeLineaire(coefficients, second_membre)
        if not systeme.solution_verifiee():
            msg = f"solution non verifiee pour le systeme {coefficients}"
            raise ValueError(msg)
        solutions.append(tuple(sp.simplify(x) for x in systeme.solution()))
    return tuple(solutions)


def determinants_des_systemes() -> tuple[sp.Expr, ...]:
    """Le determinant de chacun des trois systemes : non nul, donc solution unique."""
    return tuple(coefficients.det() for coefficients, _ in SYSTEMES)


def determinant_parametrique(numero: int) -> sp.Expr:
    """Le determinant du systeme homogene `numero` de la section 2.3."""
    return sp.expand(PARAMETRIQUES[numero].det())


def solutions_non_nulles(
    numero: int, valeurs: Mapping[sp.Symbol, sp.Expr]
) -> tuple[tuple[sp.Expr, ...], ...]:
    """Une base des solutions du systeme `numero` quand les parametres valent cela.

    Chaque vecteur est rendu sans denominateur, sa premiere composante
    non nulle positive.
    """
    noyau = PARAMETRIQUES[numero].subs(valeurs).nullspace()
    if not noyau:
        msg = f"pas de solution non nulle pour {dict(valeurs)}"
        raise ValueError(msg)
    base = []
    for vecteur in noyau:
        commun = sp.lcm([sp.fraction(sp.together(x))[1] for x in vecteur])
        entier = [sp.simplify(x * commun) for x in vecteur]
        premier = next(x for x in entier if x != 0)
        signe = -1 if premier.could_extract_minus_sign() else 1
        base.append(tuple(signe * x for x in entier))
    return tuple(base)
