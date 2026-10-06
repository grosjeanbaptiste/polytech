"""Mathematique, exercices 2, I.2 a I.5 : champ de pentes, logistique, exactes, ressort.

Source : `Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, pages 5
a 9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.application.resolue_exs_02 import X, Y
from polytech.maths.domain.differentielles.autonomes import EquationAutonome
from polytech.maths.domain.differentielles.bernoulli import EquationDeBernoulli
from polytech.maths.domain.differentielles.exactes import FormeDifferentielle
from polytech.maths.domain.differentielles.modeles import logistique
from polytech.maths.domain.differentielles.second_ordre import (
    C1,
    C2,
    EquationDuSecondOrdre,
)
from polytech.maths.domain.symboles import TEMPS as T

if TYPE_CHECKING:
    from polytech.maths.domain.differentielles.lineaire import EquationLineaire

CAPACITE: Final = sp.Symbol("K", positive=True)
TAUX: Final = sp.Symbol("lambda", positive=True)
DEPART: Final = sp.Symbol("y_0", positive=True)
LIBRE: Final = sp.Symbol("c", positive=True)
PULSATION: Final = sp.Symbol("omega", positive=True)
ALPHA: Final = sp.Symbol("alpha", real=True)
CADRE: Final = sp.Interval(-4, 4)
"""Les ordonnees que l'exercice I.2 demande de tracer."""


def champ() -> EquationAutonome:
    """I.2 : `y' = sin(pi y / 4)`."""
    return EquationAutonome(sp.sin(sp.pi * Y / 4))


def points_critiques_du_champ() -> tuple[sp.Expr, ...]:
    """Les zeros de la pente dans le cadre : autant de solutions constantes."""
    zeros = sp.solveset(champ().second_membre, Y, domain=CADRE)
    return tuple(sorted(zeros))


def isoclines(pente: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Les deux branches de `sin(pi y / 4) = p` sur une periode."""
    principale = 4 * sp.asin(pente) / sp.pi
    return principale, 4 - principale


def solution_du_champ(depart: object) -> sp.Expr:
    """La courbe integrale par `(0, y0)`, pour `-4 < y0 < 4`, `y0` non nul.

    Variables separees : `dy / sin(pi y/4) = dt` s'integre en
    `tan(pi y/8) = C e^(pi t/4)`.
    """
    constante = sp.tan(sp.pi * sp.sympify(depart) / 8)
    return 8 * sp.atan(constante * sp.exp(sp.pi * T / 4)) / sp.pi


def logistique_de_la_feuille() -> EquationAutonome:
    """I.3 : `y' = lambda y (K - y)`."""
    return logistique(TAUX, CAPACITE)


def logistique_resolue() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """La solution generale en `c`, puis `c(y0)`, puis la solution de Cauchy."""
    croissance = sp.exp(TAUX * CAPACITE * T)
    generale = CAPACITE * LIBRE * croissance / (1 + LIBRE * croissance)
    (constante,) = sp.solve(sp.Eq(generale.subs(T, 0), DEPART), LIBRE)
    return generale, constante, sp.simplify(generale.subs(LIBRE, constante))


def logistique_par_bernoulli() -> tuple[EquationLineaire, sp.Expr]:
    """`y' - lambda K y = -lambda y^2` : l'equation en `z = 1/y`, et `y(x)`."""
    equation = EquationDeBernoulli(-TAUX * CAPACITE, -TAUX, sp.Integer(2))
    return equation.equation_en_z(), equation.solution_generale()


def inflexion_de_la_logistique() -> sp.Expr:
    """L'ordonnee ou `y''` s'annule : `f'(y) = 0`, car `y'' = f'(y) y'`."""
    pente = logistique_de_la_feuille().second_membre
    (ordonnee,) = sp.solve(pente.diff(Y), Y)
    return ordonnee


def _premiere_forme(parametre: sp.Expr) -> FormeDifferentielle:
    en_x = 2 * X + parametre * Y + 2 * X * sp.exp(X**2)
    return FormeDifferentielle(en_x, 2 * Y + X)


def parametre_d_exactitude() -> sp.Expr:
    """I.4 : la valeur de `alpha` qui annule l'ecart de Schwarz."""
    (valeur,) = sp.solve(_premiere_forme(ALPHA).ecart_de_schwarz(), ALPHA)
    return valeur


def potentiels() -> tuple[sp.Expr, sp.Expr]:
    """I.4 : le potentiel de chacune des deux equations exactes."""
    premiere = _premiere_forme(parametre_d_exactitude())
    seconde = FormeDifferentielle(sp.cos(X + Y) + 2 * X, sp.cos(X + Y) + 4 * Y)
    return premiere.potentiel(), seconde.potentiel()


def ressort() -> EquationDuSecondOrdre:
    """I.5 : `y'' + y`, masse et raideur unites."""
    return EquationDuSecondOrdre(1, 0, 1)


def reponses_forcees() -> tuple[sp.Expr, sp.Expr]:
    """La solution particuliere pour `sin(omega x)`, hors resonance puis dedans."""
    hors = ressort().solution_particuliere(sp.sin(PULSATION * X))
    return hors, ressort().solution_particuliere(sp.sin(X))


def reponses_depuis_le_repos() -> tuple[sp.Expr, sp.Expr]:
    """Les memes, avec `y(0) = y'(0) = 0` : la seconde est la limite de la premiere."""
    reponses = []
    for forcee in reponses_forcees():
        generale = ressort().solution_homogene() + forcee
        conditions = (generale.subs(X, 0), generale.diff(X).subs(X, 0))
        constantes = sp.solve(conditions, (C1, C2))
        reponses.append(sp.simplify(generale.subs(constantes)))
    return reponses[0], reponses[1]
