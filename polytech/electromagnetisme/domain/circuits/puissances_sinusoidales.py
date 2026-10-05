"""Les puissances d'une branche en regime sinusoidal.

Le syllabus d'electricite generale (16.13 a 16.19). La puissance
instantanee `p = u i` vaut
`U I cos(phi) + U I cos(2 omega t + 2 theta_u - phi)` (16.13.2) : une
moyenne, la puissance **active** `P = U I cos(phi)` (16.13.4), et un
terme fluctuant de pulsation double. `Q = U I sin(phi)` (16.15.2) obeit
comme `P` a un principe de conservation : c'est la puissance
**reactive**, positive si le courant est en arriere sur la tension. Pour
une branche RLC elle compare les energies moyennes de la bobine et du
condensateur (16.15.3). La puissance apparente `S = U I`, elle, ne se
conserve pas.

Le signe de `P` dit si la branche recoit ou fournit, celui de `Q` si elle
est inductive ou capacitive. Le courant se scinde en `I cos(phi)`, actif,
et `I sin(phi)`, reactif : a puissance active et tension imposees, les
pertes d'une ligne croissent avec le second (16.19.14).
"""

from __future__ import annotations

import math
from enum import Enum

import sympy as sp

from polytech.electromagnetisme.domain.circuits.branche_sinusoidale import (
    Nature,
    nature_de_la_branche,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


class Role(Enum):
    """Ce que le signe de la puissance active dit de la branche."""

    RECEPTEUR = "recepteur"
    GENERATEUR = "generateur"


def puissance_active(tension: Grandeur, courant: Grandeur, ecart: Grandeur) -> sp.Expr:
    """`P = U I cos(phi)` (16.13.4), en watts."""
    return tension * courant * sp.cos(ecart)


def puissance_reactive(
    tension: Grandeur, courant: Grandeur, ecart: Grandeur
) -> sp.Expr:
    """`Q = U I sin(phi)` (16.15.2), en vars."""
    return tension * courant * sp.sin(ecart)


def puissance_instantanee(
    tension: Grandeur,
    courant: Grandeur,
    phase: Grandeur,
    ecart: Grandeur,
    pulsation: Grandeur,
) -> sp.Expr:
    """`p = U I cos(phi) + U I cos(2 omega t + 2 theta_u - phi)` (16.13.2)."""
    fluctuante = sp.cos(2 * pulsation * TEMPS + 2 * phase - ecart)
    return puissance_active(tension, courant, ecart) + tension * courant * fluctuante


def composantes_de_la_puissance(
    active: Grandeur, reactive: Grandeur, phase: Grandeur, pulsation: Grandeur
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`(P, P cos 2a, Q sin 2a)` de (16.15.8), ou `a = omega t + theta_u`."""
    double = 2 * (pulsation * TEMPS + phase)
    return (sp.sympify(active), active * sp.cos(double), reactive * sp.sin(double))


def erreur_relative_maximale(periodes: float, facteur: float) -> float:
    """`(T / (t_2 - t_1)) / (2 pi |cos(phi)|)` (16.14.3), la duree en periodes."""
    if periodes <= 0:
        msg = f"duree nulle ou negative : {periodes} periode"
        raise ValueError(msg)
    if facteur == 0:
        msg = "facteur de puissance nul : la puissance moyenne l'est aussi"
        raise ValueError(msg)
    return 1 / (periodes * 2 * math.pi * abs(facteur))


def puissance_reactive_rlc(
    pulsation: Grandeur, inductance: Grandeur, capacite: Grandeur, courant: Grandeur
) -> sp.Expr:
    """`Q = 2 omega (L I^2 / 2 - Q_c^2 / (2 C))` (16.15.3), ou `I = omega Q_c`."""
    efficace = sp.sympify(courant)
    charge = efficace / pulsation
    magnetique = sp.sympify(inductance) * efficace**2 / 2
    energies = magnetique - charge**2 / (2 * sp.sympify(capacite))
    return 2 * pulsation * energies


def role_et_nature(active: float, reactive: float) -> tuple[Role, Nature]:
    """Le quadrant de la figure 16.18.1."""
    if active == 0:
        msg = "puissance active nulle : ni recepteur ni generateur"
        raise ValueError(msg)
    role = Role.RECEPTEUR if active > 0 else Role.GENERATEUR
    return role, nature_de_la_branche(reactive)


def courants_actif_et_reactif(courant: float, ecart: float) -> tuple[float, float]:
    """`(I_w, I_d) = (I cos(phi), I sin(phi))` : (16.19.4) et (16.19.5)."""
    return courant * math.cos(ecart), courant * math.sin(ecart)


def pertes_en_ligne(
    resistance: float, *, puissance: float, tension: float, facteur: float
) -> float:
    """`P_l = R I^2` (16.19.14), ou `I = P / (U cos(phi))`."""
    if not 0 < facteur <= 1:
        msg = f"facteur de puissance hors de ]0 ; 1] : {facteur}"
        raise ValueError(msg)
    return resistance * (puissance / (tension * facteur)) ** 2
