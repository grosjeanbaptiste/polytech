"""Extensions, partie I : l'algorithme simplexe sous sa forme revisee.

Les regles sont celles du chapitre II -- meme variable entrante, meme
variable sortante, donc memes bases. Seul le calcul change : a chaque
iteration on ne tire des donnees que

* la premiere ligne `B^^-1 (T^, d^)`, pour le test d'optimalite et le
  choix de la colonne entrante ;
* la colonne des seconds membres et la colonne entrante, pour le choix
  de la ligne sortante ;

puis l'inverse se met a jour par un facteur : `B^_t^-1 = E^_t B^_{t-1}^-1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.algorithme import ITERATIONS_MAXIMALES, Issue
from polytech.optimisation.domain.forme_revisee import (
    facteur_etendu,
    inverse_de_base_etendue,
    tableau_des_donnees,
)

if TYPE_CHECKING:
    from polytech.optimisation.domain.formes import FormeStandard


@dataclass(frozen=True, slots=True)
class IterationRevisee:
    """Ce qu'une iteration calcule, et rien de plus."""

    base: tuple[int, ...]
    inverse: sp.Matrix
    ligne_de_z: sp.Matrix
    seconds_membres: sp.Matrix
    entrante: int | None = None
    colonne: sp.Matrix | None = None
    sortante: int | None = None
    facteur: sp.Matrix | None = None


@dataclass(frozen=True, slots=True)
class ResolutionRevisee:
    """Les iterations successives et l'arret."""

    iterations: tuple[IterationRevisee, ...]
    issue: Issue

    @property
    def valeur(self) -> sp.Expr:
        """`c_B B^-1 d`, lue en tete de la derniere ligne de `z`."""
        return self.iterations[-1].ligne_de_z[0, 0]

    @property
    def solution_duale(self) -> tuple[sp.Expr, ...]:
        """`c_B B^-1` : la premiere ligne du dernier inverse, sans son 1."""
        return tuple(self.iterations[-1].inverse[0, 1:])

    def solution(self, variables: int) -> sp.Matrix:
        """La solution de base de la derniere iteration."""
        derniere = self.iterations[-1]
        point = sp.zeros(variables, 1)
        for rang, indice in enumerate(derniere.base):
            point[indice - 1, 0] = derniere.seconds_membres[rang, 0]
        return point


def _iteration(
    donnees: sp.Matrix, base: tuple[int, ...], inverse: sp.Matrix
) -> IterationRevisee:
    """Une iteration : la ligne de `z`, puis seulement la colonne entrante."""
    ligne_de_z = inverse[0, :] * donnees[:, :-1]
    seconds = inverse[1:, :] * donnees[:, 0]
    etat = IterationRevisee(base, inverse, ligne_de_z, seconds)
    meilleur = max(ligne_de_z[0, 1:])
    if meilleur <= 0:
        return etat
    entrante = list(ligne_de_z[0, 1:]).index(meilleur) + 1
    colonne = inverse * donnees[:, entrante]
    rapports = {
        rang: seconds[rang, 0] / colonne[rang + 1, 0]
        for rang in range(len(base))
        if colonne[rang + 1, 0] > 0
    }
    if not rapports:
        return IterationRevisee(base, inverse, ligne_de_z, seconds, entrante, colonne)
    rang = min(rapports, key=lambda r: rapports[r])
    facteur = facteur_etendu(colonne, rang + 1)
    return IterationRevisee(
        base, inverse, ligne_de_z, seconds, entrante, colonne, base[rang], facteur
    )


def resoudre_en_forme_revisee(
    forme: FormeStandard,
    base: tuple[int, ...],
    iterations_maximales: int = ITERATIONS_MAXIMALES,
) -> ResolutionRevisee:
    """La phase II, en minimisation, depuis une base admissible."""
    donnees = tableau_des_donnees(forme)
    inverse = inverse_de_base_etendue(forme, base)
    if any(v < 0 for v in inverse[1:, :] * donnees[:, 0]):
        msg = f"base non admissible : {base} ; la phase II part d'un sommet"
        raise ValueError(msg)
    iterations: list[IterationRevisee] = []
    for _ in range(iterations_maximales):
        courante = _iteration(donnees, base, inverse)
        iterations.append(courante)
        if courante.entrante is None:
            return ResolutionRevisee(tuple(iterations), Issue.OPTIMALE)
        if courante.sortante is None or courante.facteur is None:
            return ResolutionRevisee(tuple(iterations), Issue.NON_BORNEE)
        rang = base.index(courante.sortante)
        base = (*base[:rang], courante.entrante, *base[rang + 1 :])
        inverse = courante.facteur * inverse
    msg = "aucun arret de la forme revisee : cyclage ?"
    raise RuntimeError(msg)
