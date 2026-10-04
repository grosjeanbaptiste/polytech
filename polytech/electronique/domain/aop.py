"""L'amplificateur operationnel, de son gain fini au zero virtuel.

Le cours resout un montage a AOP en posant `V+ = V-` : le zero virtuel.
Ce module ne le pose pas. Il modele l'AOP par sa sortie,
`A (V+ - V-)`, derriere une resistance de sortie, resout le circuit
pour un gain `A` fini, puis fait tendre `A` vers l'infini. Le zero
virtuel en *sort* -- et avec lui sa condition : qu'il y ait une
contre-reaction, faute de quoi la limite n'existe pas.

La saturation est l'autre regime : la sortie collee a un rail se
comporte comme une source de tension, et le zero virtuel ne tient plus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.electronique.domain.limites import limite_a_l_infini
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceCommandee,
    SourceDeTension,
    resoudre,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

GAIN: Final = sp.Symbol("A", positive=True)
"""Le gain en boucle ouverte, qui tend vers l'infini."""

CONDUCTANCE: Final = sp.Symbol("G_{aop}", positive=True)
"""La conductance de sortie de l'AOP, que la limite fait disparaitre."""

_COURANT_DE_SORTIE: Final = sp.Symbol("I_{sat}")


def aop(plus: int, moins: int, sortie: int) -> tuple[Element, Element]:
    """L'AOP en Norton : `A G (V+ - V-)` injecte en sortie, sur `G`.

    A vide, la sortie vaut `A (V+ - V-)`. La conductance `G` disparait
    dans la limite : seule compte la pente infinie de la caracteristique.
    """
    return (
        SourceCommandee(0, sortie, GAIN * CONDUCTANCE, plus, moins),
        Admittance(sortie, 0, CONDUCTANCE),
    )


def sortie_ideale(elements: Sequence[Element], noeuds: int, noeud: int) -> sp.Expr:
    """La tension du noeud quand le gain de chaque AOP tend vers l'infini."""
    return limite_a_l_infini(resoudre(elements, noeuds)[noeud - 1], GAIN)


def sortie_saturee(
    elements: Sequence[Element],
    noeuds: int,
    sortie: int,
    rail: sp.Expr,
) -> sp.Matrix:
    """Les tensions du circuit quand la sortie de l'AOP est collee au rail.

    L'AOP est retire du circuit et remplace par une source de tension
    ideale a sa sortie : il n'y a plus de gain, plus de zero virtuel.
    """
    sans_aop = [
        e
        for e in elements
        if not (isinstance(e, SourceCommandee) and e.arrivee == sortie)
        and e != Admittance(sortie, 0, CONDUCTANCE)
    ]
    return resoudre(
        [*sans_aop, SourceDeTension(sortie, 0, rail, _COURANT_DE_SORTIE)], noeuds
    )


def saturer(valeur: float, alimentation: float) -> float:
    """La sortie bornee aux rails `+-alimentation`."""
    if alimentation <= 0:
        msg = f"alimentation non positive : {alimentation}"
        raise ValueError(msg)
    return max(-alimentation, min(alimentation, valeur))
