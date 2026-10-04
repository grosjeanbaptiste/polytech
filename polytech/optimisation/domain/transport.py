"""Chapitre I.2.1, exemple 3 : le probleme de transport.

`r` depots de stocks `q_i` approvisionnent `s` centres de consommation
de demandes `d_j`, au cout unitaire `c_ij`. Les `r s` variables `x_ij`
sont rangees depot par depot. Le modele general ecrit `r + s`
contraintes : `somme_j x_ij <= q_i` et `somme_i x_ij >= d_j`.

Sa **matrice technologique** a une structure particuliere : chaque
variable n'intervient que deux fois, avec un coefficient `1` -- dans la
contrainte de son depot et dans celle de son centre.

Le probleme n'a de solution que si `somme(q_i) >= somme(d_j)`. Il n'est
pas restrictif d'imposer alors des egalites, en creant un **centre de
consommation fictif** qui absorbe le surplus a un cout nul.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import sympy as sp

from polytech.optimisation.domain.contraintes import (
    Contrainte,
    Optimisation,
    Sens,
    exacts,
)
from polytech.optimisation.domain.formulation import ProgrammeGeneral


@dataclass(frozen=True, slots=True)
class ProblemeDeTransport:
    """Les stocks des depots, les demandes des centres, les couts unitaires."""

    stocks: tuple[sp.Expr, ...]
    demandes: tuple[sp.Expr, ...]
    couts: tuple[tuple[sp.Expr, ...], ...]

    def __post_init__(self) -> None:
        """Rend les nombres exacts et refuse une matrice mal dimensionnee."""
        object.__setattr__(self, "stocks", exacts(self.stocks))
        object.__setattr__(self, "demandes", exacts(self.demandes))
        object.__setattr__(self, "couts", tuple(exacts(ligne) for ligne in self.couts))
        r, s = len(self.stocks), len(self.demandes)
        if len(self.couts) != r or any(len(ligne) != s for ligne in self.couts):
            msg = f"matrice des couts attendue de dimension {r} x {s}"
            raise ValueError(msg)

    @property
    def surplus(self) -> sp.Expr:
        """`somme(q_i) - somme(d_j)` : ce que les depots ont en trop."""
        return sp.Add(*self.stocks) - sp.Add(*self.demandes)

    def matrice_technologique(self) -> sp.Matrix:
        """`T`, de dimension `(r + s) x (r s)` : deux `1` par colonne."""
        r, s = len(self.stocks), len(self.demandes)
        matrice = sp.zeros(r + s, r * s)
        for i in range(r):
            for j in range(s):
                matrice[i, i * s + j] = 1
                matrice[r + j, i * s + j] = 1
        return matrice

    def programme(self) -> ProgrammeGeneral:
        """Le modele general : `<=` pour les depots, `>=` pour les centres."""
        return _ecrire(self, Sens.INFERIEUR, Sens.SUPERIEUR)

    def equilibre(self) -> ProblemeDeTransport:
        """Le meme probleme, un centre fictif absorbant le surplus a cout nul."""
        if self.surplus < 0:
            msg = f"demande totale superieure aux stocks : il manque {-self.surplus}"
            raise ValueError(msg)
        if self.surplus == 0:
            return self
        return replace(
            self,
            demandes=(*self.demandes, self.surplus),
            couts=tuple((*ligne, sp.Integer(0)) for ligne in self.couts),
        )

    def programme_equilibre(self) -> ProgrammeGeneral:
        """Le probleme equilibre, ecrit avec des egalites."""
        return _ecrire(self.equilibre(), Sens.EGAL, Sens.EGAL)


def _ecrire(
    probleme: ProblemeDeTransport,
    depots: Sens,
    centres: Sens,
) -> ProgrammeGeneral:
    """Le programme lineaire, les depots et les centres dans le sens demande."""
    matrice = probleme.matrice_technologique()
    seconds = (*probleme.stocks, *probleme.demandes)
    sens = (
        *([depots] * len(probleme.stocks)),
        *([centres] * len(probleme.demandes)),
    )
    contraintes = tuple(
        Contrainte(tuple(matrice[k, :]), sens[k], seconds[k])
        for k in range(matrice.rows)
    )
    objectif = tuple(c for ligne in probleme.couts for c in ligne)
    return ProgrammeGeneral(Optimisation.MIN, objectif, contraintes)
