"""Examen de janvier 2022 : une integrale, un systeme, une equation.

Source : `Mathematique/EXAMENS/Exam_2022_Janvier_Enonce.pdf`. Les formules
du sujet ne se lisent que sur l'image du PDF ; elles sont recopiees ici.
Le sujet n'a pas de corrige au depot.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.algebre.discussion_par_le_rang import discuter_par_le_rang
from polytech.maths.domain.algebre.solutions import solution_generale
from polytech.maths.domain.algebre.systemes import SystemeLineaire
from polytech.maths.domain.integration.definie import IntegraleDefinie
from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T
from polytech.maths.domain.symboles import VARIABLE as X

PARAMETRE: Final = sp.Symbol("m", real=True)
SYSTEME: Final = SystemeLineaire(
    sp.Matrix([[1, 1, -1], [3, 1, -1], [1, -2, 2]]),
    sp.Matrix([1, 1, PARAMETRE]),
)
PROBLEME: Final = ProblemeLineaire((1, -4, 3), T**2, (0, 0))


def integrale() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """L'etape par parties, la primitive, et la valeur entre 1 et 2."""
    integrande = sp.log(1 + X) / X**2
    parties = IntegrationParParties(sp.log(1 + X), 1 / X**2)
    definie = IntegraleDefinie(integrande, sp.Integer(1), sp.Integer(2))
    valeur = sp.expand_log(definie.valeur(), force=True)
    return parties.etape(), definie.primitive(), valeur


def systeme() -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """La seule valeur du parametre qui rend le systeme compatible, et sa droite."""
    discussion = discuter_par_le_rang(SYSTEME, PARAMETRE)
    ((valeur, _),) = discussion.particulieres.items()
    return valeur, solution_generale(discussion.en(valeur))


def equation() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """L'image de la solution, ses fractions simples, et la solution."""
    image = PROBLEME.image()
    return image, sp.apart(image, P), PROBLEME.solution()
