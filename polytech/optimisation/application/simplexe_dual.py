"""Exercice III du jeu « algorithme simplexe » : le dual du probleme de l'exercice I.

Le probleme dual, les valeurs de ses variables, et le theoreme des ecarts
complementaires qui les relie a la solution primale.
"""

from __future__ import annotations

import sympy as sp

from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.application.simplexe_bornes import (
    BASES_FINALES,
    PROGRAMME,
    M,
    forme,
    optimum,
)
from polytech.optimisation.domain.dual import dual_de
from polytech.optimisation.domain.ecarts_complementaires import (
    couples_de_contraintes,
    duale_par_les_ecarts,
)
from polytech.optimisation.domain.tableau import tableau_de_la_base

Ligne = tuple[tuple[sp.Expr, ...], str, sp.Expr]
PREMIER_ECART, PREMIERE_ARTIFICIELLE = 4, 8
"""Les colonnes de `x1e` et de `x1a` dans la forme standard des planches."""


def dual() -> tuple[tuple[sp.Expr, ...], tuple[Ligne, ...], set[int]]:
    """L'objectif, les contraintes et les variables libres (de 1) du dual."""
    probleme = dual_de(PROGRAMME)
    lignes = tuple(
        (c.coefficients, c.sens.value, c.second_membre) for c in probleme.contraintes
    )
    return probleme.objectif, lignes, {j + 1 for j in probleme.libres}


def variables_duales() -> tuple[sp.Expr, ...]:
    """La solution optimale du dual."""
    return optimum_fini(dual_de(PROGRAMME))[1]


def valeur_du_dual() -> sp.Expr:
    """`w`, qui doit egaler `z`."""
    return optimum_fini(dual_de(PROGRAMME))[0]


def produits_des_ecarts() -> tuple[sp.Expr, ...]:
    """Les `n + m` produits du theoreme : variable fois ecart de sa contrainte."""
    _, point = optimum()
    couples = couples_de_contraintes(PROGRAMME, point, variables_duales())
    return tuple(sp.simplify(c.ecart_primal * c.ecart_dual) for c in couples)


def duales_par_les_ecarts() -> tuple[sp.Expr, ...]:
    """`u`, deduite de la seule solution primale par les ecarts complementaires."""
    _, point = optimum()
    return duale_par_les_ecarts(PROGRAMME, point)


def couts_reduits_des_ecarts() -> tuple[sp.Expr, sp.Expr]:
    """`u1` et `u2`, lus dans la premiere ligne du tableau optimal.

    Pour un ecart hors base, `a_0j = -u_j` ; pour une artificielle de cout
    `M`, `a_0j = u_j - M`.
    """
    tableau = tableau_de_la_base(forme(), BASES_FINALES[-1])
    ecart = tableau.coefficients[0, tableau.hors_base.index(PREMIER_ECART) + 1]
    artificielle = tableau.coefficients[
        0, tableau.hors_base.index(PREMIERE_ARTIFICIELLE) + 1
    ]
    return -ecart, sp.expand(artificielle + M)
