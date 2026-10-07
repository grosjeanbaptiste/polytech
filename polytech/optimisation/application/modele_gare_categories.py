"""Modelisation, probleme IIIc : trente personnes en quatre categories.

Cinq ne travaillent que de jour, cinq que de nuit, cinq a cheval ; les
quinze autres font deux periodes de quatre heures **non consecutives**,
payees 10 % de plus. `s_ij` compte ceux qui travaillent aux periodes
`i` et `j`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.optimisation.application.modele_gare import (
    BESOINS,
    NUIT,
    ORDRE_DES_LIGNES,
    PERIODES,
    SALAIRES,
    en_entiers,
    ligne_de_presence,
)
from polytech.optimisation.application.modele_gare import presents as des_equipes
from polytech.optimisation.domain.contraintes import Contrainte, Optimisation, Sens
from polytech.optimisation.domain.formulation import ProgrammeGeneral

if TYPE_CHECKING:
    from collections.abc import Sequence

Paire = tuple[int, int]
PAIRES: Final[tuple[Paire, ...]] = (
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 5),
    (3, 6),
    (4, 6),
)
"""Planche 18 : `s13` a `s46`, dans l'ordre de la planche."""
MAJORATION: Final = sp.Rational(11, 10)
COUT_IMPRIME: Final = {(2, 4): sp.Rational(133, 10)}
"""La planche ecrit `13.3 s24` ; le calcul donne 13,2."""
LIGNE_IMPRIMEE_DE_LA_PERIODE_1: Final[tuple[Paire, ...]] = ((2, 4), (2, 5), (2, 6))
"""La derniere ligne de la planche repete les horaires de la periode 2."""
PLAFONDS: Final = (((3, 4), 5), ((1, 6), 5), ((2, 5), 5))
"""Jour, nuit, a cheval : les equipes de chaque categorie, et son effectif."""
COUPES: Final = 15
CONTRE_EXEMPLE: Final = (0, 4, 0, 5, 1, 0, 0, 0, 0, 0, 3, 1, 6, 0, 3)
"""Admissible pour le modele imprime : personne ne travaille de 0 h a 4 h."""


def cout_d_un_horaire_coupe(premiere: int, seconde: int) -> sp.Expr:
    """Deux periodes, la nuit comptant double, le tout majore de 10 %."""
    heures = sum(8 if p in NUIT else 4 for p in (premiere, seconde))
    return MAJORATION * heures


COUTS: Final = (*SALAIRES, *(cout_d_un_horaire_coupe(*p) for p in PAIRES))


def horaires_de_la_periode(periode: int) -> tuple[Paire, ...]:
    """Les horaires coupes qui comprennent cette periode."""
    return tuple(paire for paire in PAIRES if periode in paire)


def presents(plan: Sequence[sp.Expr | int]) -> tuple[sp.Expr | int, ...]:
    """Le nombre de personnes au travail pendant chaque periode."""
    coupes = dict(zip(PAIRES, plan[PERIODES:], strict=True))
    return tuple(
        base + sum(coupes[paire] for paire in horaires_de_la_periode(p))
        for p, base in enumerate(des_equipes(plan[:PERIODES]), start=1)
    )


def cout(plan: Sequence[sp.Expr | int]) -> sp.Expr:
    """Le cout salarial du plan, aux couts recalcules."""
    return sum((c * n for c, n in zip(COUTS, plan, strict=True)), sp.Integer(0))


def _ligne(equipes: Sequence[int], paires: Sequence[Paire]) -> tuple[int, ...]:
    """Une ligne : ces equipes (de 1) et ces horaires coupes a 1, le reste a 0."""
    debuts = tuple(int(i in equipes) for i in range(1, PERIODES + 1))
    return (*debuts, *(int(paire in paires) for paire in PAIRES))


def programme(*, corrige: bool) -> ProgrammeGeneral:
    """Le modele de la planche 18, tel qu'imprime ou sa derniere ligne corrigee."""
    return programme_des_besoins(BESOINS, corrige=corrige)


def programme_des_besoins(
    besoins: Sequence[int], *, corrige: bool = True
) -> ProgrammeGeneral:
    """Le meme modele pour d'autres besoins."""
    contraintes = []
    for p in ORDRE_DES_LIGNES:
        imprimee = p == 1 and not corrige
        paires = (
            LIGNE_IMPRIMEE_DE_LA_PERIODE_1 if imprimee else horaires_de_la_periode(p)
        )
        equipes = [i for i, pris in enumerate(ligne_de_presence(p), start=1) if pris]
        ligne = _ligne(equipes, paires)
        contraintes.append(Contrainte(ligne, Sens.SUPERIEUR, besoins[p - 1]))
    for equipes_de_la_categorie, effectif in PLAFONDS:
        ligne = _ligne(equipes_de_la_categorie, ())
        contraintes.append(Contrainte(ligne, Sens.INFERIEUR, effectif))
    contraintes.append(Contrainte(_ligne((), PAIRES), Sens.INFERIEUR, COUPES))
    return ProgrammeGeneral(Optimisation.MIN, COUTS, tuple(contraintes))


def respecte(modele: ProgrammeGeneral, plan: Sequence[int]) -> bool:
    """Vrai si le plan satisfait toutes les contraintes du modele."""
    return all(c.est_satisfaite(plan) for c in modele.contraintes)


def optimum() -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """Le cout minimal du modele corrige, et un plan qui l'atteint."""
    return en_entiers(programme(corrige=True))
