"""Les raisonnements du chapitre I, etape par etape.

`formes.forme_standard` rend le resultat ; ce module rend le **chemin**,
dans l'ordre ou le cours le parcourt pages 24 et 25 : les variables de
signe quelconque d'abord, le sens de l'optimisation ensuite, les
variables d'ecart enfin -- une par inegalite, ajoutee ou retranchee selon
le sens, et aucune pour une egalite.

Chaque etape porte sa justification et sa page : un corrige peut la
citer telle quelle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.optimisation.domain.contraintes import Optimisation, Sens
from polytech.raisonnement.domain.etapes import Etape, Raisonnement

if TYPE_CHECKING:
    from polytech.optimisation.domain.formulation import ProgrammeGeneral

SOURCE: Final = (
    "Optimisation/SLIDES/Slides_01_Les_bases_de_la_programmation_lineaire.pdf"
)
_PAGE_24: Final = f"{SOURCE}, p. 24"
_PAGE_25: Final = f"{SOURCE}, p. 25"


def _variables_libres(programme: ProgrammeGeneral) -> list[Etape]:
    etapes = []
    for j in sorted(programme.libres):
        x = f"x_{j + 1}"
        etapes.append(
            Etape(
                f"{x} = {x}^+ - {x}^-, avec {x}^+ >= 0 et {x}^- >= 0",
                "une variable de signe quelconque est la difference de deux "
                "variables non negatives",
                _PAGE_24,
            )
        )
    return etapes


def _minimisation(programme: ProgrammeGeneral) -> list[Etape]:
    if programme.optimisation is Optimisation.MIN:
        return []
    return [
        Etape(
            "max z devient min -z",
            "maximiser la fonction z est equivalent a minimiser la fonction -z",
            _PAGE_24,
        )
    ]


def _ecarts(programme: ProgrammeGeneral) -> list[Etape]:
    etapes = []
    rang = len(programme.objectif) + len(programme.libres)
    gestes = {Sens.INFERIEUR: ("+", "ajoute"), Sens.SUPERIEUR: ("-", "retranche")}
    for numero, contrainte in enumerate(programme.contraintes, start=1):
        if contrainte.sens not in gestes:
            continue
        rang += 1
        signe, geste = gestes[contrainte.sens]
        etapes.append(
            Etape(
                f"contrainte {numero} : t_{numero} x {signe} x_{rang} = d_{numero}",
                f"on {geste} une variable d'ecart non negative, qui mesure "
                "l'ecart positif entre les deux membres",
                _PAGE_25,
            )
        )
    return etapes


def expliquer_la_forme_standard(programme: ProgrammeGeneral) -> Raisonnement:
    """Le passage a `min z = c x`, `T x = d`, `x >= 0`, et pourquoi chaque pas."""
    ecarts = _ecarts(programme)
    etapes = [*_variables_libres(programme), *_minimisation(programme), *ecarts]
    if not etapes:
        etapes = [
            Etape(
                "le programme est deja sous forme standard",
                "on minimise, toutes les contraintes sont des egalites et "
                "toutes les variables sont non negatives",
                _PAGE_25,
            )
        ]
    total = len(programme.objectif) + len(programme.libres) + len(ecarts)
    return Raisonnement(
        question="Comment mettre ce programme sous forme standard ?",
        etapes=tuple(etapes),
        conclusion=(
            f"min z = c x, T x = d, x >= 0, a {total} variables "
            f"dont {len(ecarts)} variables d'ecart"
        ),
    )
