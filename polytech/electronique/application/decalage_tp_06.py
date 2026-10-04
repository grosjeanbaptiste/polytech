"""TP 6 d'electronique, exercices 5 et 7 : la tension de decalage en sortie.

Les defauts de l'AOP sont des sources ajoutees au montage, comme le fait
l'officiel : une tension `e0` en serie avec l'entree `+`, et deux courants
`I+`, `I-` que les entrees tirent vers l'AOP. Le montage est resolu tel
quel, AOP de gain fini puis gain infini ; aucune superposition n'est
ecrite a la main.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electronique.domain.aop import aop, sortie_ideale
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceDeCourant,
    SourceDeTension,
)

_I: Final = sp.Symbol("I_{e0}")


def courants_d_entree(polarisation: float, decalage: float) -> tuple[float, float]:
    """`I+` et `I-` tires des deux grandeurs que la fiche technique publie."""
    return polarisation + decalage / 2, polarisation - decalage / 2


def _exact(valeur: float) -> sp.Expr:
    return sp.nsimplify(valeur, rational=True)


def _non_inverseur(e0: float, i_plus: float, i_moins: float) -> float:
    """Exercice 5 : `Rs = 10 k` vers la source eteinte, `R1 = 1 k`, `R2 = 9 k`."""
    circuit: list[Element] = [
        Admittance(1, 0, _exact(1 / 10e3)),
        SourceDeTension(2, 1, _exact(e0), _I),
        SourceDeCourant(2, 0, _exact(i_plus)),
        Admittance(3, 0, _exact(1 / 1e3)),
        Admittance(3, 4, _exact(1 / 9e3)),
        SourceDeCourant(3, 0, _exact(i_moins)),
        *aop(2, 3, 4),
    ]
    return float(sortie_ideale(circuit, 4, 4))


def decalage_lm741() -> float:
    """`e0 = 1 mV`, `Ibias = 80 nA`, `Ioffset = 20 nA` (valeurs typiques)."""
    return _non_inverseur(1e-3, *courants_d_entree(80e-9, 20e-9))


def decalage_lm741_par_source() -> tuple[float, float, float]:
    """La part de `e0`, de `I+` et de `I-` : chaque source seule allumee."""
    i_plus, i_moins = courants_d_entree(80e-9, 20e-9)
    return (
        _non_inverseur(1e-3, 0, 0),
        _non_inverseur(0, i_plus, 0),
        _non_inverseur(0, 0, i_moins),
    )


def decalage_ca3140a() -> float:
    """`e0 = 2 mV`, `I = 10 pA`, `Ioffset = 0.5 pA` (valeurs typiques)."""
    return _non_inverseur(2e-3, *courants_d_entree(10e-12, 0.5e-12))


def decalage_de_l_exercice_7() -> float:
    """L'inverseur `R1 = 10 k`, `R2 = 50 k` sur LM741, entree eteinte.

    L'entree `+` est a la masse : `I+` n'y traverse aucune resistance.
    """
    i_plus, i_moins = courants_d_entree(80e-9, 20e-9)
    circuit: list[Element] = [
        SourceDeTension(1, 0, _exact(1e-3), _I),
        SourceDeCourant(1, 0, _exact(i_plus)),
        Admittance(2, 0, _exact(1 / 10e3)),
        Admittance(2, 3, _exact(1 / 50e3)),
        SourceDeCourant(2, 0, _exact(i_moins)),
        *aop(1, 2, 3),
    ]
    return float(sortie_ideale(circuit, 3, 3))
