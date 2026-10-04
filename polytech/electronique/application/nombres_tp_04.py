"""Les valeurs numeriques du TP 4 : source commune et cascode.

Les ordres de grandeur sont ceux de l'enonce (`TP_4.pdf`, exercices 3.2 et
3.3) ; `C_ol = 1 fF` est celui du point 3.2.4. Le calcul exact et
l'approximation du corrige officiel sont rendus cote a cote : c'est leur
ecart qui dit si l'approximation est licite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.electronique.application import deductions_tp_04 as dd
from polytech.electronique.application.symboles_tp_04 import (
    COL,
    GM,
    GM1,
    GM2,
    REQ,
    RL,
    RO1,
    RO2,
    RS,
    VS,
    R,
    S,
)

SOURCE_COMMUNE: Final = {
    RS: 50,
    R: 100,
    GM: sp.Rational(1, 20),
    REQ: 10_000,
    COL: sp.Rational(1, 10**15),
}
CASCODE: Final = {
    GM1: sp.Rational(1, 50),
    GM2: sp.Rational(1, 50),
    RO1: 20_000,
    RO2: 20_000,
}


@dataclass(frozen=True)
class SourceCommune:
    """Le gain aux deux bouts du spectre, le zero et le pole, en rad/s."""

    gain_en_continu: float
    gain_a_l_infini: float
    gain_a_l_infini_approche: float
    zero: float
    pole: float
    pole_approche: float
    gain_av: float
    capacite_miller: float


@dataclass(frozen=True)
class Cascode:
    """`V_3 / V_S` sous `R_L -> 0`, le gain a vide, le facteur Miller."""

    gain_interne: float
    gain_a_vide: float
    facteur_miller: float


def _racine(polynome: sp.Expr) -> float:
    (racine,) = sp.solve(polynome, S)
    return float(racine)


def source_commune_numerique() -> SourceCommune:
    """Les nombres de l'exercice 3.2, exacts et approches."""
    transfert = dd.transfert_de_la_source_commune().subs(SOURCE_COMMUNE)
    numerateur, denominateur = sp.fraction(sp.cancel(transfert))
    v = SOURCE_COMMUNE
    av = v[GM] * v[REQ]
    inverse = 1 / v[GM]
    return SourceCommune(
        gain_en_continu=float(transfert.subs(S, 0)),
        gain_a_l_infini=float(sp.limit(transfert, S, sp.oo)),
        gain_a_l_infini_approche=float((inverse + v[R]) / (v[R] + v[RS] + inverse)),
        zero=_racine(numerateur),
        pole=_racine(denominateur),
        pole_approche=float(
            -(1 + v[R] * v[GM]) / (v[COL] * av * (v[R] + v[RS] + inverse))
        ),
        gain_av=float(av),
        capacite_miller=float(av * v[COL]),
    )


def cascode_numerique() -> Cascode:
    """Les nombres de l'exercice 3.3."""
    v3, _ = dd.tensions_de_la_cascode()
    interne = sp.limit((v3 / VS).subs(CASCODE), RL, 0)
    return Cascode(
        gain_interne=float(interne),
        gain_a_vide=float(dd.gain_de_la_cascode_a_vide().subs(CASCODE)),
        facteur_miller=float(1 - interne),
    )
