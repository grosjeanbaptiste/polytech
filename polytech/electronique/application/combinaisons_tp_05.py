"""TP 5 d'electronique, exercices 16 a 19 : plusieurs entrees, plusieurs etages.

La superposition n'est pas *appliquee* ici : le circuit est resolu avec
toutes ses sources a la fois, et le coefficient de chacune se lit sur
l'expression obtenue. C'est ce qui permet aux tests de verifier la
superposition du corrige officiel au lieu de la recopier.
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

V1, V2, VIN, I1 = sp.symbols("V_1 V_2 V_in I_1")
S: Final = sp.Symbol("s")
_IA, _IB = sp.symbols("I_a I_b")


def _r(ohms: float) -> sp.Expr:
    return 1 / sp.nsimplify(ohms)


def _coefficients(sortie: sp.Expr, *sources: sp.Symbol) -> tuple[float, ...]:
    return tuple(float(sp.diff(sortie, source)) for source in sources)


def exercice_16() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Les coefficients de `V_1` et `V_2` dans `V_o`, pour les deux montages."""
    soustracteur: list[Element] = [
        SourceDeTension(1, 0, V2, _IA),
        SourceDeTension(2, 0, V1, _IB),
        Admittance(1, 3, _r(5e3)),
        Admittance(3, 5, _r(10e3)),
        Admittance(2, 4, _r(1e3)),
        Admittance(4, 0, _r(10e3)),
        *aop(4, 3, 5),
    ]
    sommateur: list[Element] = [
        SourceDeTension(1, 0, V1, _IA),
        SourceDeTension(2, 0, V2, _IB),
        Admittance(1, 3, _r(10e3)),
        Admittance(2, 3, _r(5e3)),
        Admittance(3, 4, _r(25e3)),
        *aop(0, 3, 4),
    ]
    return (
        _coefficients(sortie_ideale(soustracteur, 5, 5), V1, V2),
        _coefficients(sortie_ideale(sommateur, 4, 4), V1, V2),
    )


def coefficients_de_l_exercice_17(r2: float) -> tuple[float, ...]:
    """`V_1` sur l'entree non inverseuse, `V_2` a travers `R1 = 1 k`."""
    circuit: list[Element] = [
        SourceDeTension(1, 0, V1, _IA),
        SourceDeTension(2, 0, V2, _IB),
        Admittance(2, 3, _r(1e3)),
        Admittance(3, 4, _r(r2)),
        *aop(1, 3, 4),
    ]
    return _coefficients(sortie_ideale(circuit, 4, 4), V1, V2)


def exercice_18() -> float:
    """`I_1` en parallele sur `R3`, puis l'inverseur `R1`, `R2`."""
    circuit: list[Element] = [
        SourceDeCourant(0, 1, sp.Rational(1, 2000)),
        Admittance(1, 0, _r(4e3)),
        Admittance(1, 2, _r(1e3)),
        Admittance(2, 3, _r(5e3)),
        *aop(0, 2, 3),
    ]
    return float(sortie_ideale(circuit, 3, 3))


def exercice_18_court_circuit() -> float:
    """Le second schema : un fil relie la source et l'entree `+` a la masse.

    La source debite dans ce fil ; elle n'a plus aucun noeud a elle, et
    l'amplificateur ne voit qu'une entree a la masse.
    """
    circuit: list[Element] = [
        SourceDeCourant(0, 0, sp.Rational(1, 2000)),
        Admittance(0, 1, _r(1e3)),
        Admittance(1, 2, _r(5e3)),
        *aop(0, 1, 2),
    ]
    return float(sortie_ideale(circuit, 2, 2))


def _montage_19() -> list[Element]:
    return [
        SourceDeTension(1, 0, VIN, _IA),
        Admittance(1, 2, _r(1e3)),
        *aop(2, 3, 3),
        Admittance(3, 4, _r(1e3)),
        Admittance(4, 5, _r(4e3)),
        SourceDeCourant(0, 6, I1),
        Admittance(6, 0, _r(1e3)),
        *aop(6, 4, 5),
        Admittance(5, 7, _r(1e3)),
        Admittance(8, 9, _r(9e3)),
        Admittance(8, 10, _r(1e3)),
        Admittance(10, 0, S * sp.Symbol("C", positive=True)),
        *aop(7, 8, 9),
    ]


def decalage_de_l_exercice_19() -> float:
    """La tension continue que `I_1` pose sur l'entree `+` du deuxieme etage."""
    tension = sortie_ideale(_montage_19(), 10, 6).subs(VIN, 0)
    return float(tension.subs(I1, sp.Rational(1, 10**6)))


def exercice_19() -> tuple[float, float]:
    """`V_o` continue (condensateur ouvert) et gain alternatif (court-circuit)."""
    sortie = sortie_ideale(_montage_19(), 10, 9)
    continu = sp.limit(sortie.subs(VIN, 0), S, 0).subs(I1, sp.Rational(1, 10**6))
    alternatif = sp.limit(sp.diff(sortie, VIN), S, sp.oo)
    return float(continu), float(alternatif)
