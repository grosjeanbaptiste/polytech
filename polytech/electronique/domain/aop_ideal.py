"""Un circuit a AOP ideaux, resolu en posant le zero virtuel.

`aop.py` modele l'AOP par un gain fini puis passe a la limite : le zero
virtuel en *sort*. Avec trois AOP et dix composants litteraux, cette
limite ne se calcule plus en un temps raisonnable. Ce module pose donc
directement ce que la limite etablit, comme le cours le fait a la main :

* les deux entrees d'un AOP sont au meme potentiel, `V+ = V-` ;
* sa sortie fournit le courant qu'il faut : elle n'a pas de loi des
  noeuds, et c'est l'egalite ci-dessus qui la remplace.

Un test confronte les deux modeles sur un meme montage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp
from sympy.polys.matrices import DomainMatrix

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from polytech.electronique.domain.netliste import Admittance

Aop = tuple[int, int, int]
"""Les noeuds `(plus, moins, sortie)` ; 0 est la masse."""


def _ligne_des_noeuds(
    noeud: int, admittances: Sequence[Admittance], noeuds: int
) -> list[sp.Expr]:
    """La loi des noeuds en `noeud` : somme des `y (V_autre - V_noeud)`."""
    ligne = [sp.Integer(0)] * (noeuds + 1)
    for branche in admittances:
        for ici, la in (
            (branche.noeud_a, branche.noeud_b),
            (branche.noeud_b, branche.noeud_a),
        ):
            if ici != noeud:
                continue
            ligne[noeud - 1] -= branche.valeur
            if la > 0:
                ligne[la - 1] += branche.valeur
    return ligne


def tensions_a_aop_ideaux(
    admittances: Sequence[Admittance],
    aops: Sequence[Aop],
    sources: Mapping[int, sp.Expr],
    noeuds: int,
) -> tuple[sp.Expr, ...]:
    """Les tensions des noeuds 1 a `noeuds`, AOP ideaux et sources imposees."""
    sorties = {sortie: (plus, moins) for plus, moins, sortie in aops}
    if len(sorties) != len(aops) or set(sorties) & set(sources):
        msg = "une sortie d'AOP est partagee, ou imposee par une source"
        raise ValueError(msg)
    lignes = []
    for noeud in range(1, noeuds + 1):
        ligne = [sp.Integer(0)] * (noeuds + 1)
        if noeud in sources:
            ligne[noeud - 1], ligne[noeuds] = sp.Integer(1), sp.sympify(sources[noeud])
        elif noeud in sorties:
            for entree, signe in zip(sorties[noeud], (1, -1), strict=True):
                if entree > 0:
                    ligne[entree - 1] += signe
        else:
            ligne = _ligne_des_noeuds(noeud, admittances, noeuds)
        lignes.append(ligne)
    augmentee = DomainMatrix.from_Matrix(sp.Matrix(lignes)).to_field()
    reduite, pivots = augmentee.rref()
    if pivots != tuple(range(noeuds)):
        msg = "circuit sans solution unique : un AOP n'a pas de contre-reaction"
        raise ValueError(msg)
    return tuple(sp.cancel(v) for v in reduite.to_Matrix()[:, -1])
