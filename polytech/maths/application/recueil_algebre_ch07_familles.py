"""Recueil d'algebre, chapitre VII : libre, generatrice, base, coordonnees.

Les familles sont dans `recueil_algebre_ch07_enonces` ; tout se ramene au
rang d'une matrice dont les lignes sont les vecteurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.maths.application import recueil_algebre_ch07_enonces as enonces
from polytech.maths.domain.algebre.familles_de_vecteurs import (
    appartient,
    coordonnees,
    est_generatrice,
    est_libre,
    est_une_base,
    extraire_une_base,
    prolonger_en_base,
    rang_de_la_famille,
    relation_de_dependance,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

Famille = enonces.Famille
Base = tuple["Sequence[object]", ...]
"""Ce que le domaine rend : des vecteurs, sans promesse sur leur type."""
TOUTES: Final = enonces.VECTEURS | enonces.MATRICES | enonces.POLYNOMES
LIBRE_OU_LIEE: Final = ("3a", "3b", "4", "23", "31")
BASES_A_VERIFIER: Final = ("9", "12", "13a", "13b", "34")
LIEES: Final = ("4", "10F1", "23", "25", "26", "31", "32F1", "35", "37")
"""Les familles liees dont le corrige ecrit la relation de dependance."""
CIBLES: Final = enonces.CIBLES | {"34": (1, 1, 1)}
"""Au 34, le polynome `1 + X + X^2` a ecrire dans la base."""


@dataclass(frozen=True, slots=True)
class Nature:
    """Ce que les exercices 10 et 32 demandent d'une famille."""

    libre: bool
    generatrice: bool
    extraite: Base
    completee: Base

    @property
    def base(self) -> bool:
        """Libre et generatrice a la fois."""
        return self.libre and self.generatrice


def famille(cle: str) -> Famille:
    """La famille de cette cle ; leve si le chapitre n'en a pas."""
    if cle not in TOUTES:
        msg = f"pas de famille {cle!r} au chapitre VII"
        raise KeyError(msg)
    return TOUTES[cle]


def libertes() -> dict[str, bool]:
    """Exercices 3, 4, 23 et 31 : vrai si la famille est libre."""
    return {cle: est_libre(famille(cle)) for cle in LIBRE_OU_LIEE}


def relations() -> dict[str, tuple[sp.Expr, ...]]:
    """Pour chaque famille liee, des coefficients d'une combinaison nulle."""
    return {cle: relation_de_dependance(famille(cle)) for cle in LIEES}


def condition_du_5() -> sp.Expr:
    """Le determinant des trois vecteurs : ils sont libres ou il n'est pas nul."""
    return sp.factor(sp.Matrix(famille("5")).det())


def determinants() -> dict[str, sp.Expr]:
    """Le determinant de chaque famille carree que le corrige calcule a la main."""
    cles = ("3a", "3b", *BASES_A_VERIFIER)
    return {cle: sp.simplify(sp.Matrix(famille(cle)).det()) for cle in sorted(cles)}


def coordonnees_demandees() -> dict[str, tuple[sp.Expr, ...]]:
    """Exercices 6, 9, 13 et 34 : le vecteur dans la famille donnee."""
    return {
        cle: coordonnees(v, famille(cle)) for cle, v in CIBLES.items() if cle != "7"
    }


def sont_des_bases() -> dict[str, bool]:
    """Exercices 9, 12, 13 et 34 : vrai si la famille est une base."""
    return {cle: est_une_base(famille(cle)) for cle in BASES_A_VERIFIER}


def exercice_7() -> bool:
    """Vrai si `(0, 0, 1)` est dans le sous-espace engendre."""
    return appartient(CIBLES["7"], famille("7"))


def exercice_8() -> int:
    """Le rang de la famille."""
    return rang_de_la_famille(famille("8"))


def _nature(vecteurs: Famille) -> Nature:
    libre = est_libre(vecteurs)
    completee: Base = prolonger_en_base(vecteurs) if libre else ()
    return Nature(
        libre, est_generatrice(vecteurs), extraire_une_base(vecteurs), completee
    )


def natures() -> dict[str, Nature]:
    """Exercices 10 et 32 : libre, generatrice, base extraite ou completee."""
    cles = [cle for cle in TOUTES if cle.startswith(("10F", "32F"))]
    return {cle: _nature(famille(cle)) for cle in cles}
