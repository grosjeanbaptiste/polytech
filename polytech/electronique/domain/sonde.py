"""La masse d'une sonde d'oscilloscope : elle relie un noeud a la terre.

Le Picoscope partage sa masse avec son generateur. Brancher sa borne
negative sur un noeud qui n'est pas la masse du montage fusionne ce noeud
avec la masse : tout ce qui s'y trouve branche est court-circuite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.electronique.domain.aop import saturer
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceDeTension,
    resoudre,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_COURANT = sp.Symbol("I_source")
_ENTREE = 1


def _fusionne(noeud: int, masse: int) -> int:
    """Le numero du noeud apres fusion de `masse` avec la terre."""
    if noeud == masse:
        return 0
    return noeud - 1 if noeud > masse else noeud


def _tensions(branches: Sequence[Admittance], noeuds: int) -> sp.Matrix:
    source = SourceDeTension(_ENTREE, 0, sp.Integer(1), _COURANT)
    elements: list[Element] = [source, *branches]
    return resoudre(elements, noeuds)


def transfert_sans_sonde(
    branches: Sequence[Admittance], noeuds: int, mesure: tuple[int, int]
) -> sp.Expr:
    """`(V_plus - V_moins) / V_entree`, la sonde ne perturbant rien."""
    tensions = _tensions(branches, noeuds)
    plus, moins = mesure
    return sp.simplify(tensions[plus - 1] - tensions[moins - 1])


def transfert_avec_la_masse_de_la_sonde(
    branches: Sequence[Admittance],
    noeuds: int,
    mesure: tuple[int, int],
    masse: int,
) -> sp.Expr:
    """Ce que la sonde lit quand sa masse est sur le noeud `masse`."""
    if masse == _ENTREE:
        msg = "la masse de la sonde sur l'entree annulerait la source"
        raise ValueError(msg)
    fusionnees = [
        Admittance(_fusionne(b.noeud_a, masse), _fusionne(b.noeud_b, masse), b.valeur)
        for b in branches
    ]
    tensions = _tensions(fusionnees, noeuds - 1)
    plus, moins = (_fusionne(n, masse) for n in mesure)
    lues = [sp.Integer(0), *tensions[: noeuds - 1]]
    return sp.simplify(lues[plus] - lues[moins])


def sortie_du_non_inverseur_sonde_a_la_masse(
    entree: float, gain: float, alimentation: float
) -> float:
    """`V_- = 0` : plus de contre-reaction, `V_o = sat(A_v V_in)`."""
    return saturer(gain * entree, alimentation)
