"""Les circuits de la seance 3.2 en regime transitoire : exercices 2 et 7.

Les deux exercices commutent un circuit **alimente en sinusoidal**, et
c'est ce qui les distingue de la seance 3.1. La condition initiale ne
vaut plus zero ou une valeur de regime continu : elle est la valeur d'une
sinusoide a l'instant du basculement, donc elle depend de la phase.

- Exercice 2 (figure 3.3.14) : le condensateur, en regime sinusoidal
  derriere `R_0`, bascule a l'instant zero sur `R` suivi de `R_1` en
  parallele avec `R_2`-`L_2`. Le courant `i` sort du condensateur par sa
  borne chargee positivement.
- Exercice 7 (figure 3.3.19) : un `R`-`L` au repos enclenche sur
  `e(t) = E racine 2 cos(omega t + theta_u)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.electromagnetisme.application import symboles_seance_32 as d
from polytech.electromagnetisme.domain.circuits.elements_operationnels import (
    Bobine,
    Capacite,
)
from polytech.electromagnetisme.domain.circuits.enclenchement import enclenchement
from polytech.electromagnetisme.domain.circuits.operationnel import (
    ReseauOperationnel,
)
from polytech.electromagnetisme.domain.circuits.phaseurs import (
    en_regime_sinusoidal,
    valeur_instantanee,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Resistance,
    SourceDeTension,
)
from polytech.electromagnetisme.domain.symboles import PULSATION, TEMPS

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.circuits.elements_operationnels import (
        Element,
    )


def circuit_du_deux_avant() -> ReseauOperationnel:
    """Position 1 : la source sinusoidale charge `C` a travers `R_0`."""
    return ReseauOperationnel(
        elements=(
            SourceDeTension("s", "masse", d.E0 * sp.exp(sp.I * d.THETA0)),
            Resistance("s", "c", d.R0),
            Capacite("c", "masse", d.C),
        ),
        reference="masse",
    )


def tension_initiale_du_deux() -> sp.Expr:
    """`u_C(0)` : le regime sinusoidal de la position 1, lu a l'instant zero."""
    phaseur = en_regime_sinusoidal(circuit_du_deux_avant().potentiels()["c"], PULSATION)
    return valeur_instantanee(phaseur, PULSATION).subs(TEMPS, 0)


def circuit_du_deux_apres() -> ReseauOperationnel:
    """Position 2 : `C`, charge a `u_C(0)`, se decharge dans le reste."""
    elements: tuple[Element, ...] = (
        Capacite("c", "masse", d.C, tension_initiale=d.U0),
        Resistance("c", "a", d.R),
        Resistance("a", "masse", d.R1),
        Resistance("a", "b", d.R2),
        Bobine("b", "masse", d.L2),
    )
    return ReseauOperationnel(elements=elements, reference="masse")


def courant_du_deux() -> sp.Expr:
    """`I(p)`, le courant qui traverse `R` depuis le condensateur."""
    potentiels = circuit_du_deux_apres().potentiels()
    return sp.factor((potentiels["c"] - potentiels["a"]) / d.R)


BOBINE_DU_SEPT = Bobine("b", "masse", d.L)


def circuit_du_sept() -> ReseauOperationnel:
    """Figure 3.3.19 en regime etabli : la source, `R` et `L` en serie."""
    return ReseauOperationnel(
        elements=(
            SourceDeTension("s", "masse", d.E * sp.exp(sp.I * d.THETA)),
            Resistance("s", "b", d.R),
            BOBINE_DU_SEPT,
        ),
        reference="masse",
    )


def regime_du_sept() -> sp.Expr:
    """Le courant force, une fois le transitoire eteint."""
    phaseur = en_regime_sinusoidal(circuit_du_sept().courant(BOBINE_DU_SEPT), PULSATION)
    return valeur_instantanee(phaseur, PULSATION)


def courant_du_sept() -> sp.Expr:
    """Le regime force, raccorde a `i(0) = 0` par l'exponentielle."""
    return enclenchement(regime_du_sept(), sp.Integer(0), d.L / d.R)
