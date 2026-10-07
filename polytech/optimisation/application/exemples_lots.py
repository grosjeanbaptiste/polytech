"""Exemples resolus, planches 79 a 85 : quatre ecritures du « lot-sizing ».

Cinq mois, une demande par mois ; produire un mois coute un forfait `f`
et un cout unitaire `p`, stocker d'un mois au suivant coute `h`. Les
quatre ecritures ont le meme optimum entier ; leurs relaxations, non, et
c'est ce qui decide de la taille de l'arborescence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.optimisation.application.exemples_lots_fermeture import (
    DEMANDES,
    FORFAITS,
    STOCKAGES,
    UNITAIRES,
    fermeture_convexe,
)
from polytech.optimisation.application.optimum import optimum_fini
from polytech.optimisation.domain.branch_and_bound import (
    resoudre_en_variables_entieres,
)
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.entiers import ProgrammeEntier
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    import sympy as sp

MOIS: Final = len(DEMANDES)
GRAND_M: Final = (100,) * MOIS
"""Ecriture I (planche 81) : un majorant grossier de la production d'un mois."""
RESTES: Final = tuple(sum(DEMANDES[i:]) for i in range(MOIS))
"""Ecriture II (planche 82) : on ne produit jamais plus que la demande a venir."""
_DEBUTS: Final = {"x": 0, "s": MOIS, "y": 2 * MOIS - 1}
_VARIABLES: Final = 3 * MOIS - 1
OBJECTIF: Final = (*UNITAIRES, *STOCKAGES, *FORFAITS)


def _ligne(**termes: int) -> tuple[int, ...]:
    """Une ligne de coefficients, les variables nommees `x1`, `s3`, `y2`..."""
    ligne = [0] * _VARIABLES
    for nom, coefficient in termes.items():
        ligne[_DEBUTS[nom[0]] + int(nom[1:]) - 1] = coefficient
    return tuple(ligne)


def _bilans() -> list[Contrainte]:
    """`s_(i-1) + x_i - s_i = d_i` : ce qui entre dans le mois en sort."""
    lignes = []
    for mois, demande in enumerate(DEMANDES, start=1):
        termes = {f"x{mois}": 1}
        if mois > 1:
            termes[f"s{mois - 1}"] = 1
        if mois < MOIS:
            termes[f"s{mois}"] = -1
        lignes.append(Contrainte(_ligne(**termes), Sens.EGAL, demande))
    return lignes


def _ouvertures(majorants: tuple[int, ...]) -> list[Contrainte]:
    """`x_i <= M_i y_i` et `y_i <= 1` : produire exige d'avoir ouvert le mois."""
    liens = [
        Contrainte(_ligne(**{f"x{i}": 1, f"y{i}": -m}), Sens.INFERIEUR, 0)
        for i, m in enumerate(majorants, start=1)
    ]
    unites = [
        Contrainte(_ligne(**{f"y{i}": 1}), Sens.INFERIEUR, 1)
        for i in range(1, MOIS + 1)
    ]
    return liens + unites


def _renforts() -> list[Contrainte]:
    """Ecriture III (planche 83) : `x_i - s_i <= d_i y_i`, la part servie au mois."""
    return [
        Contrainte(
            _ligne(**{f"x{i}": 1, f"s{i}": -1, f"y{i}": -DEMANDES[i - 1]}),
            Sens.INFERIEUR,
            0,
        )
        for i in range(2, MOIS)
    ]


def ecriture(nom: str) -> ProgrammeGeneral:
    """Le programme de l'ecriture I, II, III ou IV, les `y` relaches."""
    if nom == "IV":
        return fermeture_convexe()
    premier = Contrainte(_ligne(y1=1), Sens.EGAL, 1)
    contraintes = {
        "I": _bilans() + _ouvertures(GRAND_M),
        "II": [*_bilans(), *_ouvertures(RESTES), premier],
        "III": [*_bilans(), *_ouvertures(RESTES), premier, *_renforts()],
    }
    return ProgrammeGeneral(Optimisation.MIN, OBJECTIF, tuple(contraintes[nom]))


def relaxation(nom: str) -> sp.Expr:
    """La valeur de la relaxation lineaire de cette ecriture."""
    return optimum_fini(ecriture(nom))[0]


def ouvertures_de_la_relaxation(nom: str) -> tuple[sp.Expr, ...]:
    """Les `y` a l'optimum de la relaxation : les cinq dernieres variables."""
    return optimum_fini(ecriture(nom))[1][-MOIS:]


def sommets_explores(nom: str) -> int:
    """Le nombre de sommets de l'arborescence, les `y` etant binaires."""
    programme = ecriture(nom)
    binaires = frozenset(range(_VARIABLES - MOIS, _VARIABLES))
    exploration = resoudre_en_variables_entieres(ProgrammeEntier(programme, binaires))
    return len(exploration.noeuds)
