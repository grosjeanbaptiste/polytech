"""La separation et l'evaluation : le branch and bound du chapitre XIV.

Le chapitre en donne un exemple dont la structure est lisible sur la
planche : la relaxation vaut `v_0 = 45/2`, on branche sur `x_3 <= 1` et
`x_3 >= 2`, et l'une des deux branches donne `v_1 = 23` -- entiere, donc
**sondee**, c'est-a-dire qu'elle devient la meilleure solution connue et
que toute branche dont la relaxation ne la depasse pas est elaguee.

Trois gestes, et chacun est un objet ici : **relaxer**, **separer**,
**evaluer**. L'elagage est ce qui distingue l'algorithme d'une enumeration
exhaustive, et c'est pourquoi le nombre de noeuds explores est rendu : un
corrige doit pouvoir montrer que l'arbre a ete coupe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp

from polytech.optimisation.domain.forme_standard import ProgrammeLineaire
from polytech.optimisation.domain.simplexe import Simplexe

NOEUDS_MAXIMAUX = 500
"""Garde-fou : un arbre non elague peut exploser combinatoirement."""


def _est_entier(valeur: sp.Expr) -> bool:
    return bool(sp.simplify(valeur - sp.floor(valeur)) == 0)


def premiere_fractionnaire(point: sp.Matrix) -> int | None:
    """L'indice de la premiere composante non entiere, ou rien si toutes le sont."""
    return next(
        (rang for rang, valeur in enumerate(point) if not _est_entier(valeur)),
        None,
    )


def separer(
    programme: ProgrammeLineaire,
    variable: int,
    seuil: int,
    *,
    vers_le_haut: bool,
) -> ProgrammeLineaire:
    """Ajoute `x_i <= seuil` ou `-x_i <= -seuil` selon le sens du branchement."""
    ligne = sp.zeros(1, programme.variables)
    ligne[0, variable] = -1 if vers_le_haut else 1
    borne = sp.Matrix([-seuil if vers_le_haut else seuil])
    return ProgrammeLineaire(
        objectif=programme.objectif,
        contraintes=programme.contraintes.col_join(ligne),
        seconds_membres=programme.seconds_membres.col_join(borne),
    )


@dataclass(frozen=True, slots=True)
class Resolution:
    """Le resultat du parcours : le meilleur point entier et l'effort fourni."""

    point: sp.Matrix | None
    valeur: sp.Expr | None
    noeuds: int


def resoudre_en_entiers(programme: ProgrammeLineaire) -> Resolution:
    """Separation et evaluation, avec elagage par la meilleure borne connue."""
    attente = [programme]
    meilleur: sp.Matrix | None = None
    valeur: sp.Expr | None = None
    noeuds = 0
    while attente:
        noeuds += 1
        if noeuds > NOEUDS_MAXIMAUX:
            msg = f"arbre de separation non elague : plus de {NOEUDS_MAXIMAUX} noeuds"
            raise ValueError(msg)
        courant = attente.pop()
        try:
            point, borne = Simplexe(programme=courant).optimum()
        except ValueError:
            continue
        if valeur is not None and borne <= valeur:
            continue
        fractionnaire = premiere_fractionnaire(point)
        if fractionnaire is None:
            meilleur, valeur = point, borne
            continue
        seuil = math.floor(float(point[fractionnaire, 0]))
        attente.append(separer(courant, fractionnaire, seuil, vers_le_haut=False))
        attente.append(separer(courant, fractionnaire, seuil + 1, vers_le_haut=True))
    return Resolution(point=meilleur, valeur=valeur, noeuds=noeuds)
