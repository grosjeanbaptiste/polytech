"""Exercice I, planche 11 : une solution optimale « non unique et non degeneree ».

Dans le tableau final, le cout reduit de `x+` est nul. Faire entrer `x+`
en base ne change pas l'objectif : `x-` croit d'autant, et la difference
`x = x+ - x-` ne bouge pas. La solution n'est multiple que dans les
variables dedoublees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.optimisation.application.simplexe_bornes import BASES_FINALES, forme
from polytech.optimisation.domain.optimales import colonnes_des_rayons_optimaux
from polytech.optimisation.domain.tableau import TableauSimplexe, tableau_de_la_base

if TYPE_CHECKING:
    import sympy as sp

X_PLUS, X_MOINS = 1, 2
"""Les colonnes de `x+` et de `x-` dans la forme standard des planches."""
ARTIFICIELLES: Final = (8, 9)


def _final() -> TableauSimplexe:
    """Le tableau optimal, artificielles ecartees comme sur la planche 11."""
    complet = tableau_de_la_base(forme(), BASES_FINALES[-1])
    gardees = [0] + [
        rang
        for rang, indice in enumerate(complet.hors_base, start=1)
        if indice not in ARTIFICIELLES
    ]
    hors_base = tuple(i for i in complet.hors_base if i not in ARTIFICIELLES)
    return TableauSimplexe(complet.base, hors_base, complet.coefficients[:, gardees])


def cout_reduit_de_x_plus() -> sp.Expr:
    """`a_0j` de la colonne de `x+` dans le tableau optimal."""
    tableau = _final()
    return tableau.coefficients[0, tableau.hors_base.index(X_PLUS) + 1]


def x_plus_ouvre_un_rayon() -> bool:
    """Vrai si `x+` peut croitre sans fin en gardant l'objectif."""
    return X_PLUS in colonnes_des_rayons_optimaux(_final())


def deplacement_de_x_le_long_du_rayon() -> sp.Expr:
    """De combien `x = x+ - x-` varie quand `x+` croit d'une unite."""
    tableau = _final()
    colonne = tableau.hors_base.index(X_PLUS) + 1
    ligne = tableau.base.index(X_MOINS) + 1
    return 1 + tableau.coefficients[ligne, colonne]


def est_degeneree() -> bool:
    """Vrai si une variable de base est nulle a l'optimum."""
    return any(v == 0 for v in _final().coefficients[1:, 0])
