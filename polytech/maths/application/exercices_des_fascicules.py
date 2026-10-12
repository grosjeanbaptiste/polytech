r"""Les trois exercices que les fascicules posent sans les resoudre, prefixe `Fasc`.

`Slides_01_Algebre_lineaire_et_integrales.pdf`, planches 9 et 12 : deux
systemes 3 x 3. `Slides_02_Equations_differentielles.pdf`, planche 24 :
`dy/dx + 2y = 3`, « a resoudre comme exercice » par separation des
variables. La planche note que l'equation est aussi lineaire : la variation
des constantes des planches 38 a 40 sert de second chemin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.application import recueil_algebre_ch03_resolution as resolution
from polytech.maths.application.recueil_algebre import (
    ecriture,
    liste,
    nom,
    operations,
)
from polytech.maths.domain.algebre.echelonnement import echelonner
from polytech.maths.domain.differentielles.lineaire import EquationLineaire
from polytech.maths.domain.differentielles.premier_ordre import INCONNUE
from polytech.maths.domain.differentielles.separables import solutions_constantes

if TYPE_CHECKING:
    from polytech.maths.domain.algebre.solutions import Nature

PREFIXE = "Fasc"
PLANCHE_9: Final = sp.Matrix([[2, 3, -1, 3], [4, 5, 2, 8], [3, 2, -1, 0]])
PLANCHE_12: Final = sp.Matrix([[1, 2, -1, 2], [-1, -2, 1, -2], [2, 4, -2, 4]])
EN_Y_24: Final = 3 - 2 * INCONNUE
"""`dy/dx = 3 - 2y` : le second membre, fonction de `y` seul."""
LINEAIRE_24: Final = EquationLineaire(sp.Integer(2), sp.Integer(3))
"""La meme equation, lue `y' + 2y = 3`."""


def systeme_pose(augmentee: sp.Matrix) -> tuple[Nature, tuple[sp.Expr, ...]]:
    """La nature du systeme et sa solution, generale s'il est indetermine."""
    return resolution.nature(augmentee), resolution.solution(augmentee)


def primitive_en_y() -> sp.Expr:
    """`int dy / (3 - 2y)`, le membre de gauche une fois les variables separees."""
    return sp.integrate(1 / EN_Y_24, INCONNUE)


def solutions_constantes_de_la_planche_24() -> tuple[sp.Expr, ...]:
    """Les zeros de `3 - 2y`, que la division ecarte."""
    return solutions_constantes(EN_Y_24)


def solution_de_la_planche_24() -> sp.Expr:
    """La solution generale, par la variation des constantes."""
    return LINEAIRE_24.solution_generale()


def _systeme(numero: int, augmentee: sp.Matrix) -> dict[str, str]:
    echelonnee = echelonner(augmentee, colonnes=augmentee.cols - 1)
    return {
        nom(PREFIXE, numero, "Operations"): operations(echelonnee.operations),
        nom(PREFIXE, numero, "Echelonnee"): ecriture(echelonnee.matrice),
        nom(PREFIXE, numero, "Simple"): ecriture(augmentee.rref()[0]),
        nom(PREFIXE, numero, "Solution"): (
            rf"\left({liste(systeme_pose(augmentee)[1])}\right)"
        ),
    }


def _equation() -> dict[str, str]:
    (constante,) = solutions_constantes_de_la_planche_24()
    absolue = primitive_en_y().replace(sp.log, lambda a: sp.log(sp.Abs(a)))
    return {
        nom(PREFIXE, 24, "Primitive"): ecriture(absolue),
        nom(PREFIXE, 24, "Constante"): ecriture(constante),
        nom(PREFIXE, 24, "Homogene"): ecriture(LINEAIRE_24.solution_homogene()),
        nom(PREFIXE, 24, "K"): ecriture(LINEAIRE_24.constante_variable()),
        nom(PREFIXE, 24, "Particuliere"): ecriture(LINEAIRE_24.solution_particuliere()),
        nom(PREFIXE, 24, "Solution"): ecriture(solution_de_la_planche_24()),
        nom(PREFIXE, 24, "AsymptoteNum"): str(constante),
        nom(PREFIXE, 24, "TauxNum"): str(LINEAIRE_24.p),
    }


def macros_des_exercices_des_fascicules() -> dict[str, str]:
    """Toutes les formules que le corrige des trois exercices compose."""
    return {**_systeme(9, PLANCHE_9), **_systeme(12, PLANCHE_12), **_equation()}
