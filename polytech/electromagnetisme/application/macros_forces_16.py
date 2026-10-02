"""Macros de l'exercice 16 : force sur l'entrefer d'un electroaimant.

Entierement symbolique, comme l'exercice 11 : l'enonce ne donne aucun
nombre. Le jeu publie donc des **expressions**, et la chaine entiere --
champ, induction, flux, flux embrasse, inductance, energie, force -- pour
que le document puisse montrer chaque maillon.

La derniere publication est la plus utile : la force a **entrefer
double**, qui chiffre la decroissance en `1/a^2`. C'est elle qui explique
pourquoi un electroaimant est si puissant au contact et si faible a un
millimetre, et pourquoi il est difficile a asservir.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import expression_latex
from polytech.electromagnetisme.domain.magnetostatique.entrefer import (
    force_d_un_electroaimant,
)

SPIRES = sp.Symbol("N_1", positive=True)
COURANT = sp.Symbol("I", positive=True)
SECTION = sp.Symbol("S", positive=True)
ENTREFER = sp.Symbol("alpha", positive=True)
PERMEABILITE = sp.Symbol("mu_0", positive=True)


def macros_forces_seize() -> dict[str, str]:
    """La chaine entiere du raisonnement, maillon par maillon."""
    champ = SPIRES * COURANT / ENTREFER
    induction = PERMEABILITE * champ
    flux = induction * SECTION
    embrasse = SPIRES * flux
    inductance = embrasse / COURANT
    energie = inductance * COURANT**2 / 2
    force = force_d_un_electroaimant(SPIRES, COURANT, SECTION, ENTREFER)
    double = force.subs(ENTREFER, 2 * ENTREFER)
    return {
        "SeizeChamp": expression_latex(champ),
        "SeizeInduction": expression_latex(induction),
        "SeizeFlux": expression_latex(sp.simplify(flux)),
        "SeizeFluxEmbrasse": expression_latex(sp.simplify(embrasse)),
        "SeizeInductance": expression_latex(sp.simplify(inductance)),
        "SeizeEnergie": expression_latex(sp.simplify(energie)),
        "SeizeForce": expression_latex(sp.simplify(force)),
        "SeizeForceADoubleEntrefer": expression_latex(sp.simplify(double)),
        "SeizeRapportDesForces": expression_latex(
            sp.simplify(force / double),
        ),
    }
