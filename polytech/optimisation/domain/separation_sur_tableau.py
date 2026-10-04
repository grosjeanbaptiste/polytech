"""Chapitre XIV.3.2 : le processus de separation, ecrit sur le tableau simplexe.

Une variable de base `x_l = a_l0` non entiere separe le noeud en deux.
Avec `a_l0 = [a_l0] + f`, `0 < f < 1` :

a) `x_l <= [a_l0]`. La variable d'ecart `x_l^e` de cette contrainte
   remplace `x_l` dans sa ligne : `a_l0 -> -f`, `a_lj -> -a_lj` ;
b) `x_l >= [a_l0] + 1`. De meme avec `x_l'` : `a_l0 -> -(1 - f)`, les
   `a_lj` inchanges.

Dans les deux cas la base reste duale admissible et cesse d'etre primale
admissible : l'algorithme dual reoptimise, `x_l^e` ou `x_l'` sortant de
la base a la premiere iteration.

La ligne garde l'indice `l` ; une **substitution** `x_l = c + s x~_l`
(`s = +-1`) retient ce que la variable du tableau represente, et se
compose quand on separe a nouveau sur la meme variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.tableau import TableauSimplexe

if TYPE_CHECKING:
    from collections.abc import Mapping


class Branche(Enum):
    """Les deux sous-noeuds d'une separation."""

    BASSE = "a) x_l <= [a_l0]"
    HAUTE = "b) x_l >= [a_l0] + 1"


@dataclass(frozen=True, slots=True)
class NoeudTableau:
    """Un tableau duale admissible, et ce que ses variables separees representent."""

    tableau: TableauSimplexe
    substitutions: Mapping[int, tuple[sp.Expr, int]] = field(default_factory=dict)
    description: str = ""
    minorant: sp.Expr | None = None

    def valeur_d_origine(self, indice: int, valeur: sp.Expr) -> sp.Expr:
        """`x_l = c + s x~_l`, ou `x_l` lui-meme s'il n'a jamais ete separe."""
        constante, signe = self.substitutions.get(indice, (sp.Integer(0), 1))
        return constante + signe * valeur


def partie_decimale(valeur: sp.Expr) -> sp.Expr:
    """`f` tel que `a = [a] + f`, `0 <= f < 1`."""
    return valeur - sp.floor(valeur)


def _fils(noeud: NoeudTableau, indice: int, branche: Branche) -> NoeudTableau:
    tableau = noeud.tableau
    ligne = tableau.base.index(indice) + 1
    valeur = tableau.coefficients[ligne, 0]
    plancher, f = sp.floor(valeur), partie_decimale(valeur)
    constante, signe = noeud.substitutions.get(indice, (sp.Integer(0), 1))
    coefficients = tableau.coefficients.copy()
    if branche is Branche.BASSE:
        coefficients[ligne, :] = -coefficients[ligne, :]
        coefficients[ligne, 0] = -f
        nouvelle = (constante + signe * plancher, -signe)
    else:
        coefficients[ligne, 0] = -(1 - f)
        nouvelle = (constante + signe * (plancher + 1), signe)
    vers_le_bas = (branche is Branche.BASSE) == (signe == 1)
    description = f"x_{indice} {'<=' if vers_le_bas else '>='} {nouvelle[0]}"
    return NoeudTableau(
        TableauSimplexe(tableau.base, tableau.hors_base, coefficients),
        {**noeud.substitutions, indice: nouvelle},
        description,
    )


def separer_le_tableau(
    noeud: NoeudTableau,
    indice: int,
) -> tuple[NoeudTableau, NoeudTableau]:
    """Les sous-noeuds a) et b) de la separation sur la variable de base `x_l`."""
    if indice not in noeud.tableau.base:
        msg = f"x_{indice} est hors base, donc nulle : rien a separer"
        raise ValueError(msg)
    ligne = noeud.tableau.base.index(indice) + 1
    if partie_decimale(noeud.tableau.coefficients[ligne, 0]) == 0:
        msg = f"x_{indice} est deja entiere : rien a separer"
        raise ValueError(msg)
    return _fils(noeud, indice, Branche.BASSE), _fils(noeud, indice, Branche.HAUTE)
