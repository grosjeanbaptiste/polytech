"""TP 6 d'electronique, exercice 6 : l'inverseur sous alimentation unique.

Le pont `R`-`R` pose `2.5 V` sur l'entree `+` ; le condensateur `C`, en
serie avec `R1`, bloque le continu. Le montage est resolu d'un bloc,
`C` compris (admittance `s C`), puis lu aux deux bouts du spectre.
Les deux `R` du pont ne sont pas donnes : ils restent symboliques, et le
resultat n'en depend pas.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.electronique.domain.aop import aop, sortie_ideale
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceDeTension,
)

S: Final = sp.Symbol("s")
VIN: Final = sp.Symbol("V_in")
_R: Final = sp.Symbol("R", positive=True)
_IA, _IB = sp.symbols("I_a I_b")


def _sortie() -> sp.Expr:
    """`V_o` : source de `5 V` au noeud 1, `V_in` au noeud 3."""
    circuit: list[Element] = [
        SourceDeTension(1, 0, sp.Integer(5), _IA),
        Admittance(1, 2, 1 / _R),
        Admittance(2, 0, 1 / _R),
        SourceDeTension(3, 0, VIN, _IB),
        Admittance(3, 4, S * sp.Rational(80, 10**6)),
        Admittance(4, 5, sp.Rational(1, 1000)),
        Admittance(5, 6, sp.Rational(1, 100_000)),
        *aop(2, 5, 6),
    ]
    return sortie_ideale(circuit, 6, 6)


def repos_de_sortie() -> float:
    """La composante continue de `V_o` : `s -> 0`, entree eteinte."""
    return float(sp.limit(_sortie().subs(VIN, 0), S, 0))


def gain_alternatif() -> float:
    """`s -> oo` : le condensateur est un fil, l'inverseur est intact."""
    return float(sp.limit(sp.diff(_sortie(), VIN), S, sp.oo))


def module_a_vingt_hertz() -> float:
    """`|H_1(j 2 pi 20)|` avec `C = 80 uF` : le bas de la bande audio."""
    transfert = sp.diff(_sortie(), VIN)
    return float(sp.Abs(transfert.subs(S, sp.I * 2 * sp.pi * 20)))


def coupure_du_filtre_de_sortie() -> float:
    """Le passe-haut `C2 = 800 nF`, `R3 = 100 k` : son pole, en hertz."""
    sortie = sp.Symbol("V_s")
    circuit: list[Element] = [
        SourceDeTension(1, 0, sortie, _IA),
        Admittance(1, 2, S * sp.Rational(800, 10**9)),
        Admittance(2, 0, sp.Rational(1, 100_000)),
    ]
    _, denominateur = sp.fraction(sp.cancel(sortie_ideale(circuit, 2, 2) / sortie))
    (pole,) = sp.solve(denominateur, S)
    return float(abs(pole)) / (2 * math.pi)
