"""Exemples resolus, planches 45 a 62 : la dualite et l'algorithme dual.

Le dual d'un programme, ce que ses variables valent -- le prix des
ressources --, et l'algorithme qui part d'une base duale admissible.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.dual import dual_de
from polytech.optimisation.domain.formes import forme_standard
from polytech.optimisation.domain.formulation import ProgrammeGeneral
from polytech.optimisation.domain.resolution import resoudre
from polytech.optimisation.domain.tableau import tableau_de_la_base

if TYPE_CHECKING:
    from polytech.optimisation.domain.algorithme import Issue

INF, SUP, EGAL = Sens.INFERIEUR, Sens.SUPERIEUR, Sens.EGAL
Optimum = tuple[sp.Expr, tuple[sp.Expr, ...]]
ILLUSTRATION: Final = ProgrammeGeneral(
    Optimisation.MAX,
    (1, -4),
    (
        Contrainte((1, 0), INF, 7),
        Contrainte((-3, 2), INF, -4),
        Contrainte((1, 1), EGAL, 5),
    ),
    libres=frozenset({1}),
)
BUREAUX: Final = ProgrammeGeneral(
    Optimisation.MAX,
    (150, 100),
    (
        Contrainte((1, 1), INF, 100),
        Contrainte((2, 1), INF, 150),
        Contrainte((5, 3), INF, 350),
    ),
)
"""Tablettes, tiroirs, heures de travail (planche 51)."""


def _a_deux_contraintes(*objectif: int) -> ProgrammeGeneral:
    contraintes = (Contrainte((1, 1, 1), SUP, 6), Contrainte((2, 0, -2), SUP, 9))
    return ProgrammeGeneral(Optimisation.MIN, objectif, contraintes)


PROGRAMMES: Final = {
    "dual": _a_deux_contraintes(6, 3, 2),
    "artificielle I": _a_deux_contraintes(6, 3, -2),
    "artificielle II": _a_deux_contraintes(-6, 3, -2),
}
"""Planches 54, 56 et 59 : trois objectifs pour les memes contraintes."""
BASES_DUALES: Final = ((4, 5), (4, 1), (2, 1))
"""Planche 55 : les bases des trois tableaux de l'algorithme dual."""


def dual_de_l_illustration() -> tuple[
    tuple[sp.Expr, ...], tuple[object, ...], set[int]
]:
    """L'objectif, les contraintes et les variables libres (de 1) du dual."""
    probleme = dual_de(ILLUSTRATION)
    lignes = tuple(
        (c.coefficients, c.sens.value, c.second_membre) for c in probleme.contraintes
    )
    return probleme.objectif, lignes, {j + 1 for j in probleme.libres}


def optimum_des_bureaux() -> Optimum:
    """Les quantites a produire et le gain total."""
    return optimum_fini(BUREAUX)


def prix_des_ressources() -> Optimum:
    """L'optimum du dual : un prix par ressource."""
    return optimum_fini(dual_de(BUREAUX))


def restes_des_ressources() -> tuple[sp.Expr, ...]:
    """Ce qu'il reste de chaque ressource a l'optimum."""
    _, point = optimum_des_bureaux()
    return tuple(c.ecart(point) for c in BUREAUX.contraintes)


def gain_d_une_unite_de_plus(ressource: int) -> sp.Expr:
    """Ce que le gain total gagne si cette ressource (de 1) croit d'une unite."""
    contraintes = list(BUREAUX.contraintes)
    visee = contraintes[ressource - 1]
    contraintes[ressource - 1] = replace(visee, second_membre=visee.second_membre + 1)
    elargi = replace(BUREAUX, contraintes=tuple(contraintes))
    return optimum_fini(elargi)[0] - optimum_des_bureaux()[0]


def cout_en_ressources(consommation: tuple[int, ...]) -> sp.Expr:
    """Ce que vaut, aux prix duaux, ce qu'un produit consomme."""
    _, prix = prix_des_ressources()
    return sum((p * c for p, c in zip(prix, consommation, strict=True)), sp.Integer(0))


def entre_en_fabrication(*, gain: int, consommation: tuple[int, ...]) -> bool:
    """Vrai si le gain du nouveau produit depasse le prix de ce qu'il consomme."""
    return bool(gain > cout_en_ressources(consommation))


def optimum(nom: str) -> Optimum:
    """La valeur et le point optimaux du programme de ce nom."""
    return optimum_fini(PROGRAMMES[nom])


def issue(nom: str) -> Issue:
    """Ce que le simplexe conclut sur le programme de ce nom."""
    return resoudre(PROGRAMMES[nom]).issue


def parcours_dual() -> tuple[tuple[sp.Expr, bool], ...]:
    """La valeur de chaque tableau de la planche 55, et s'il est admissible."""
    standard = forme_standard(PROGRAMMES["dual"])
    tableaux = [tableau_de_la_base(standard, base) for base in BASES_DUALES]
    return tuple(
        (t.valeur, all(v >= 0 for v in t.coefficients[1:, 0])) for t in tableaux
    )


def duales_par_le_dual() -> tuple[sp.Expr, ...]:
    """Les variables duales de l'exemple de la planche 54, par son dual."""
    return optimum_fini(dual_de(PROGRAMMES["dual"]))[1]


def duales_lues_dans_le_tableau() -> tuple[sp.Expr, ...]:
    """Les memes, lues dans la premiere ligne du tableau optimal : `u_j = -a_0j`."""
    final = tableau_de_la_base(forme_standard(PROGRAMMES["dual"]), BASES_DUALES[-1])
    premiere = dict(zip(final.hors_base, final.coefficients[0, 1:], strict=True))
    return tuple(-premiere[ecart] for ecart in BASES_DUALES[0])
