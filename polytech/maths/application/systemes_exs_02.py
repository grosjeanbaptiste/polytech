"""Mathematique, exercices 2, seance III : cinq systemes differentiels.

Source : `Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, pages 17
a 19.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.differentielles.plan_de_phase import (
    Nature,
    nature_du_point_critique,
)
from polytech.maths.domain.differentielles.systemes import SystemeDifferentiel
from polytech.maths.domain.differentielles.wronskien import determinant_par_la_trace
from polytech.maths.domain.laplace.proprietes import de_la_derivee_n
from polytech.maths.domain.laplace.systemes import resoudre_le_systeme
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import TEMPS as T

C1, C2 = sp.symbols("C_1 C_2", real=True)
INITIALES: Final = sp.Matrix([C1, C2])
PREMIER: Final = SystemeDifferentiel(sp.Matrix([[-3, sp.sqrt(2)], [sp.sqrt(2), -2]]))
DEUXIEME: Final = SystemeDifferentiel(sp.Matrix([[1, -1], [-4, 1]]))
TROISIEME: Final = SystemeDifferentiel(sp.Matrix([[-7, 1], [-2, -5]]))
INCONNUES: Final = (sp.Function("x")(T), sp.Function("y")(T))


def _a_la_plus_petite_composante_unite(vecteur: sp.Matrix) -> sp.Matrix:
    """Le meme vecteur propre, dans l'echelle ou la feuille l'ecrit."""
    echelle = min((sp.Abs(c) for c in vecteur), key=float)
    return sp.simplify(vecteur / echelle)


def modes_du_1() -> tuple[tuple[sp.Expr, sp.Matrix], ...]:
    """III.1 : les couples `(r, xi)`, le mode lent d'abord."""
    modes = (
        (valeur, _a_la_plus_petite_composante_unite(vecteur))
        for valeur, vecteur in PREMIER.modes_propres()
    )
    return tuple(sorted(modes, key=lambda mode: -mode[0]))


def trajectoire_du_1(premiere: object, seconde: object) -> sp.Matrix:
    """`c1 xi1 e^(r1 t) + c2 xi2 e^(r2 t)`."""
    (r1, xi1), (r2, xi2) = modes_du_1()
    return premiere * xi1 * sp.exp(r1 * T) + seconde * xi2 * sp.exp(r2 * T)


def trajectoire_depuis(position: sp.Matrix) -> sp.Matrix:
    """La solution qui passe par `position` en `t = 0`."""
    (_, xi1), (_, xi2) = modes_du_1()
    premiere, seconde = sp.Matrix.hstack(xi1, xi2).solve(position)
    return trajectoire_du_1(premiere, seconde)


def wronskiens_du_1() -> tuple[sp.Expr, sp.Expr]:
    """Le determinant des deux solutions, puis le meme par la trace de `A`."""
    colonnes = [trajectoire_du_1(1, 0), trajectoire_du_1(0, 1)]
    wronskien = PREMIER.wronskien(sp.Matrix.hstack(*colonnes))
    initial = wronskien.subs(T, 0)
    return wronskien, determinant_par_la_trace(PREMIER.matrice, t0=0, initial=initial)


def nature_du_1() -> Nature:
    """Le portrait de phase autour de l'origine."""
    return nature_du_point_critique(PREMIER.matrice)


def solutions_du_2() -> tuple[sp.Matrix, sp.Matrix]:
    """III.2 : par la transformee de Laplace, puis par l'exponentielle de matrice."""
    return DEUXIEME.par_laplace(INITIALES), DEUXIEME.par_exponentielle(INITIALES)


def resolvante_du_2() -> sp.Matrix:
    """`(pI - A)^-1` pour le systeme III.2."""
    return DEUXIEME.resolvante()


def solution_du_3() -> sp.Matrix:
    """III.3 : valeurs propres complexes, solution reelle."""
    return TROISIEME.par_laplace(INITIALES)


def valeurs_propres_du_3() -> tuple[sp.Expr, ...]:
    """`-6 + i` et `-6 - i`."""
    return tuple(valeur for valeur, _ in TROISIEME.modes_propres())


def _quatrieme() -> tuple[sp.Eq, sp.Eq]:
    x, y = INCONNUES
    return (
        sp.Eq(x.diff(T, 2) + y.diff(T) + x, sp.exp(T)),
        sp.Eq(x.diff(T) + y.diff(T, 2), 1),
    )


def solution_du_4() -> tuple[sp.Expr, sp.Expr]:
    """III.4 : le systeme du second ordre, au repos."""
    x, y = INCONNUES
    originaux = resoudre_le_systeme(_quatrieme(), {x: (0, 0), y: (0, 0)})
    return originaux[x], originaux[y]


def images_du_4() -> tuple[sp.Expr, sp.Expr]:
    """III.4 : `X(p)` et `Y(p)`, solutions du systeme algebrique."""
    image_x, image_y = sp.symbols("X Y")
    repos = (0, 0)
    equations = (
        de_la_derivee_n(image_x, repos)
        + de_la_derivee_n(image_y, repos[:1])
        + image_x
        - Transformee.de(sp.exp(T)).image,
        de_la_derivee_n(image_x, repos[:1])
        + de_la_derivee_n(image_y, repos)
        - Transformee.de(sp.Integer(1)).image,
    )
    resolues = sp.solve(equations, (image_x, image_y))
    return sp.simplify(resolues[image_x]), sp.simplify(resolues[image_y])


def solution_du_5() -> tuple[sp.Expr, sp.Expr]:
    """III.5 : `x' + x + 3y' + 7y = H(t)`, `-2x + y' + y = e^(-t)`, au repos."""
    x, y = INCONNUES
    equations = (
        sp.Eq(x.diff(T) + x + 3 * y.diff(T) + 7 * y, 1),
        sp.Eq(-2 * x + y.diff(T) + y, sp.exp(-T)),
    )
    originaux = resoudre_le_systeme(equations, {x: (0,), y: (0,)})
    return originaux[x], originaux[y]


def permanent_du_5() -> tuple[sp.Expr, sp.Expr]:
    """Ce vers quoi `x` et `y` tendent."""
    x, y = solution_du_5()
    return sp.limit(x, T, sp.oo), sp.limit(y, T, sp.oo)


def natures() -> tuple[Nature, Nature, Nature]:
    """Le portrait de phase de chacun des trois systemes homogenes."""
    systemes = (PREMIER, DEUXIEME, TROISIEME)
    premier, deuxieme, troisieme = (
        nature_du_point_critique(systeme.matrice) for systeme in systemes
    )
    return premier, deuxieme, troisieme


def retour_a_l_origine_du_2() -> sp.Expr:
    """III.2 : la valeur de `C2` qui eteint le mode divergent `e^(3t)`."""
    composante = sp.expand(DEUXIEME.par_laplace(INITIALES)[0])
    (valeur,) = sp.solve(composante.coeff(sp.exp(3 * T)), C2)
    return valeur
