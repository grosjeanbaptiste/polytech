"""La fermeture d'un circuit RL ou RC sur une source constante.

Le syllabus d'electricite generale (15.8 a 15.11). Le courant dans une
resistance ne pouvant etre infini, la charge d'un condensateur (15.8.1)
et le courant d'une bobine (15.8.2) ne varient pas brusquement : ce sont
les conditions initiales.

Bobine : `E = R i + L di/dt` (15.9.1) donne
`i = (E / R)(1 - exp(-t / tau))` (15.9.3), `tau = L / R`. Condensateur :
`E = R dq/dt + q / C` (15.10.2) donne `q = C E - (C E - Q_0) exp(-t / tau)`
(15.10.10), `tau = R C`.

La solution particuliere est le phenomene de **regime** ; la solution de
l'equation homogene, qui s'eteint, le phenomene **transitoire**.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from polytech.electromagnetisme.domain.circuits.transitoires import (
    reponse_du_premier_ordre,
)
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE


def courant_rl(fem: sp.Expr, resistance: sp.Expr, inductance: sp.Expr) -> sp.Expr:
    """`i = (E / R)(1 - exp(-t / tau))` (15.9.3), `tau = L / R`."""
    regime = fem / resistance
    return reponse_du_premier_ordre(sp.Integer(0), regime, inductance / resistance)


def transformee_du_courant_rl(
    fem: sp.Expr, resistance: sp.Expr, inductance: sp.Expr
) -> sp.Expr:
    """`I(p) = E / (p (p L + R))` (15.9.9), le courant partant de zero."""
    return fem / (VARIABLE * (VARIABLE * inductance + resistance))


def charge_rc(
    fem: sp.Expr,
    resistance: sp.Expr,
    capacite: sp.Expr,
    *,
    initiale: Any = 0,  # noqa: ANN401 -- reel ou symbole
) -> sp.Expr:
    """`q = C E - (C E - Q_0) exp(-t / tau)` (15.10.10), `tau = R C`."""
    depart = sp.sympify(initiale)
    return reponse_du_premier_ordre(depart, capacite * fem, resistance * capacite)


def transformee_de_la_charge_rc(
    fem: sp.Expr, resistance: sp.Expr, capacite: sp.Expr
) -> sp.Expr:
    """`Q(p) = C E / (p (p R C + 1))` (15.10.8), le condensateur partant vide."""
    return capacite * fem / (VARIABLE * (VARIABLE * resistance * capacite + 1))


def regime_et_transitoire(reponse: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`(regime, transitoire)` : ce qui reste a temps infini, et ce qui s'eteint."""
    regime = sp.limit(reponse, TEMPS, sp.oo)
    return regime, sp.simplify(reponse - regime)
