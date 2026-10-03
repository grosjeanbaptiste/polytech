"""Le circuit du laboratoire II d'electromagnetisme : un R-C serie.

Source : `Electromagnetisme/LABOS/Lab_02_Circuit_RC_Enonce.pdf`. Le
materiel est fixe par l'enonce -- `R = 10 kohms` pour la charge,
`2,2 kohms` pour le regime sinusoidal, `C = 0,01 uF` -- et la sortie est
prise sur le condensateur.

Tout passe par le resolveur en domaine de Laplace : la charge sous un
echelon, l'impulsion de duree `D` -- dont la transformee porte le retard
`e^{-pD}` --, et le regime sinusoidal, restriction a `p = j omega`.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.circuits.elements_operationnels import Capacite
from polytech.electromagnetisme.domain.circuits.operationnel import ReseauOperationnel
from polytech.electromagnetisme.domain.circuits.phaseurs import en_regime_sinusoidal
from polytech.electromagnetisme.domain.circuits.reseau import (
    Resistance,
    SourceDeTension,
)
from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE, original

RESISTANCE_DE_CHARGE: Final = 10_000
RESISTANCE_SINUS: Final = 2200
CAPACITE: Final = 1e-8
FREQUENCE: Final = 2000
"""Le generateur sinusoidal : `1 cos(2 pi 2000 t)`, un volt, sans continu."""


def _rc(source: sp.Expr, resistance: sp.Expr, capacite: sp.Expr) -> sp.Expr:
    """La transformee de la tension du condensateur sous cette source."""
    reseau = ReseauOperationnel(
        elements=(
            SourceDeTension("s", "m", source),
            Resistance("s", "c", resistance),
            Capacite("c", "m", capacite),
        ),
        reference="m",
    )
    return reseau.potentiels()["c"]


def charge_du_condensateur(
    v0: sp.Expr, resistance: sp.Expr, capacite: sp.Expr
) -> sp.Expr:
    """`v_C(t)` sous l'echelon `v0`, condensateur decharge."""
    return sp.simplify(original(_rc(v0 / VARIABLE, resistance, capacite)))


def constante_de_temps() -> float:
    """`tau = R C` du montage de la partie 1, en secondes."""
    return RESISTANCE_DE_CHARGE * CAPACITE


def reponse_a_l_impulsion(tau: sp.Expr, duree: sp.Expr) -> sp.Expr:
    """`v_C / V0` sous une impulsion de duree `D`, par la transformee de Laplace."""
    impulsion = (1 - sp.exp(-VARIABLE * duree)) / VARIABLE
    return original(_rc(impulsion, tau, 1))


def phaseurs_du_sinus() -> tuple[complex, complex]:
    """`V_R` et `V_C` pour une source de phaseur un, a 2 kHz.

    `V_R` est tire du courant du condensateur, `j omega C V_C`, qui
    traverse aussi la resistance -- et non de `V_s - V_C`. Que leur somme
    redonne la source est donc une verification de la loi des mailles,
    pas une consequence de l'ecriture.
    """
    pulsation = 2 * sp.pi * FREQUENCE
    condensateur = en_regime_sinusoidal(_rc(1, RESISTANCE_SINUS, CAPACITE), pulsation)
    courant = sp.I * pulsation * CAPACITE * condensateur
    return complex(sp.N(RESISTANCE_SINUS * courant)), complex(sp.N(condensateur))


LECTURES_OFFICIELLES: Final = {
    "montee": 208e-6,
    "descente": 204e-6,
    "premier_instant": 7.015e-3,
    "premiere_tension": 4.3,
    "second_instant": 7.24e-3,
    "seconde_tension": 0.52,
    "decalage_c": 23e-6,
    "decalage_r": 101e-6,
}
"""Les mesures que porte le corrige officiel, pages 4 a 6.

Ce sont des lectures d'oscilloscope, non des calculs : le corrige les
reprend pour montrer ce qu'elles donnent, et qu'elles ne s'accordent
qu'a quelques pour cent pres -- l'ordre de la tolerance des composants.
"""
