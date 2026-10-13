"""Trois calculs que les notes manuscrites de 2011 laissent en suspens.

`Syllabus_Notes_de_cours_2011.pdf` s'ouvre sur des pages ecrites a la main
(pages 1 a 47 du PDF). Trois calculs y sont annonces et non menes : un
determinant a verifier (page 14), une integration par parties repetee
arretee sur des points de suspension (page 44), et une primitive renvoyee
a « un exercice » (page 47).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.application.exercices_des_syllabus import EXEMPLE_13
from polytech.maths.application.primitives_a_exs_01 import parties
from polytech.maths.application.recueil_primitives_gestes import U, changement
from polytech.maths.application.recueil_primitives_sommes import fractions_simples
from polytech.maths.domain.symboles import VARIABLE as X

if TYPE_CHECKING:
    from polytech.maths.application.resolution_exs_01 import Resolution

POLYNOME_44: Final = X**2 + 3 * X + 2
TRINOME_47: Final = X**2 - X + 1
FRACTION_47: Final = (3 * X**2 + 1) / (X**3 + 1)


def determinants_de_la_page_14() -> tuple[sp.Expr, sp.Expr]:
    """`det A`, puis le determinant ou `b` remplace la troisieme colonne."""
    coefficients = EXEMPLE_13[:, :3]
    borde = sp.Matrix.hstack(EXEMPLE_13[:, :2], EXEMPLE_13[:, 3])
    return coefficients.det(), borde.det()


def rangs_de_la_page_14() -> tuple[int, int]:
    """Le rang de `A`, et celui de `A` completee de la colonne `b`."""
    return EXEMPLE_13[:, :3].rank(), EXEMPLE_13.rank()


def parties_de_la_page_44() -> Resolution:
    """`(x^2 + 3x + 2) e^x` : deux integrations par parties, `u` etant le polynome."""
    return parties(POLYNOME_44, sp.exp(X), gestes=2)


def fraction_de_la_page_47() -> tuple[Resolution, sp.Expr]:
    """La primitive de `1 / (x^2 - x + 1)`, puis celle de `(3x^2 + 1) / (x^3 + 1)`."""
    centre = (2 * X - 1) / sp.sqrt(3)
    en_u = 2 / (sp.sqrt(3) * (1 + U**2))
    arc = changement(1 / TRINOME_47, en_u, centre)
    return arc, fractions_simples(FRACTION_47).primitive


def coefficients_de_la_page_46() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`A`, `B`, `C` de `A / (x + 1) + (Bx + C) / (x^2 - x + 1)`."""
    a, b, c = sp.symbols("A B C")
    numerateur = a * TRINOME_47 + (b * X + c) * (X + 1) - (3 * X**2 + 1)
    solution = sp.solve(sp.Poly(numerateur, X).all_coeffs(), [a, b, c])
    return solution[a], solution[b], solution[c]


def reste_du_numerateur() -> sp.Expr:
    """`5x - 1 = 5/2 (2x - 1) + reste` : ce qui reste apres la derivee du trinome."""
    _, b, c = coefficients_de_la_page_46()
    numerateur = 3 * (b * X + c)
    return sp.expand(numerateur - numerateur.coeff(X) / 2 * sp.diff(TRINOME_47, X))
