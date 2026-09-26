"""Cours II.3 -- le U inverse d'un manometre.

Entre deux points d'un meme fluide, la pression ne depend que de la
difference d'altitude (module ``equilibre``). Une bosse -- on monte de ``b``,
on redescend de ``b + h`` -- voit donc sa hauteur propre disparaitre de
l'equation : seule subsiste la denivelee nette ``h``. C'est la seule vraie
subtilite de l'exercice, et elle se demontre en deux lignes.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.statique.symboles import RHO, G

HAUTEUR_BOSSE = sp.Symbol("b", positive=True)
"""Hauteur de la bosse : arbitraire, et c'est tout l'interet."""


def contributions_de_la_bosse(denivelee: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Contributions de la montee puis de la redescente, avec leurs signes."""
    montee = -RHO * G * HAUTEUR_BOSSE
    descente = RHO * G * (HAUTEUR_BOSSE + denivelee)
    return (montee, descente)


def compensation_de_la_bosse(denivelee: sp.Expr) -> sp.Eq:
    """Montre que ``b`` s'elimine : la bosse ne laisse que la denivelee."""
    montee, descente = contributions_de_la_bosse(denivelee)
    return sp.Eq(montee + descente, sp.expand(montee + descente), evaluate=False)
