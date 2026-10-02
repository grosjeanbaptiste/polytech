"""Macros de l'exercice 5 de statique : la charge devant un plan.

Comme l'exercice 11, celui-ci est **entierement litteral** : l'enonce ne
donne aucun nombre, et le jeu publie donc des expressions LaTeX et non
des valeurs. Il n'y a pas d'application numerique pour rattraper une
erreur de signe, ce qui rend la verification symbolique d'autant plus
necessaire.

Le prefixe `StatiqueCinq` isole ces macros de celles des autres
exercices -- un test du registre refuse toute collision de nom.

Les quatre dernieres macros publient ce que le corrige manuscrit ne
calcule pas : la charge induite totale, la force d'attraction, l'energie
d'interaction -- dont la derivee redonne la force, ce qui les verifie
l'une par l'autre -- et le coefficient dipolaire de la decroissance au
loin.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import expression_latex
from polytech.electromagnetisme.domain.electrostatique.images import (
    PERMITTIVITE,
    champ_sur_le_plan,
    charge_induite_totale,
    densite_induite,
    energie_d_interaction,
    force_sur_la_charge,
)

CHARGE = sp.Symbol("q", positive=True)
DISTANCE = sp.Symbol("a", positive=True)
ORDONNEE, COTE = sp.symbols("y z", real=True)
RAYON = sp.Symbol("r", positive=True)


def _rayon() -> sp.Expr:
    """`sqrt(a^2 + y^2 + z^2)`, la distance commune aux deux charges.

    Commune, et c'est tout l'exercice : c'est son egalite pour la charge
    et pour son image qui fait s'annuler les composantes tangentielles.
    """
    return sp.sqrt(DISTANCE**2 + ORDONNEE**2 + COTE**2)


def _coefficient_dipolaire() -> sp.Expr:
    """La limite de `r^3 E_x` au loin : le moment du dipole, au signe pres.

    Publiee pour que le document puisse montrer **pourquoi** la
    decroissance est en `1/r^3` plutot que l'affirmer.
    """
    champ_x = champ_sur_le_plan(CHARGE, DISTANCE, RAYON, sp.Integer(0))[0]
    return sp.limit(champ_x * RAYON**3, RAYON, sp.oo)


def macros_statique_cinq() -> dict[str, str]:
    """Le champ, la densite, et les trois ajouts au corrige officiel."""
    champ = champ_sur_le_plan(CHARGE, DISTANCE, ORDONNEE, COTE)
    au_pied = densite_induite(CHARGE, DISTANCE, sp.Integer(0), sp.Integer(0))
    return {
        "StatiqueCinqRayon": expression_latex(_rayon()),
        "StatiqueCinqChampX": expression_latex(sp.simplify(champ[0])),
        "StatiqueCinqChampY": expression_latex(sp.simplify(champ[1])),
        "StatiqueCinqChampZ": expression_latex(sp.simplify(champ[2])),
        "StatiqueCinqDensite": expression_latex(
            sp.simplify(densite_induite(CHARGE, DISTANCE, ORDONNEE, COTE)),
        ),
        "StatiqueCinqDensiteAuPied": expression_latex(sp.simplify(au_pied)),
        "StatiqueCinqPermittivite": expression_latex(PERMITTIVITE),
        "StatiqueCinqChargeInduite": expression_latex(
            charge_induite_totale(CHARGE, DISTANCE),
        ),
        "StatiqueCinqForce": expression_latex(force_sur_la_charge(CHARGE, DISTANCE)),
        "StatiqueCinqEnergie": expression_latex(
            energie_d_interaction(CHARGE, DISTANCE),
        ),
        "StatiqueCinqCoefficientDipolaire": expression_latex(
            _coefficient_dipolaire(),
        ),
    }
