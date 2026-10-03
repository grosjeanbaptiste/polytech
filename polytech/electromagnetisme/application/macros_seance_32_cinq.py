"""Macros de la seance 3.2, exercice 5 : la puissance d'une branche active.

La puissance de la branche, puis sa decomposition entre la source et
l'impedance `R_3 // C_3`. La somme des deux redonne la premiere : c'est
le second chemin, et il dit au passage que la source `e_3` absorbe.

Les deux aides de conversion servent aussi a l'exercice 1.

Le prefixe est `TroisDeuxCinq`.
"""

from __future__ import annotations

import cmath
import math

import sympy as sp

from polytech.electromagnetisme.application import symboles_seance_32 as d
from polytech.electromagnetisme.application.enonces_seance_32 import (
    branche_du_cinq,
    impedance_du_cinq,
)
from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.domain.circuits.phaseurs import puissance_complexe


def complexe(expression: sp.Expr) -> complex:
    """La valeur numerique d'un phaseur exact."""
    return complex(sp.N(expression, 30))


def degres(valeur: complex) -> float:
    """L'argument d'un phaseur, en degres."""
    return math.degrees(cmath.phase(valeur))


def macros_du_cinq() -> dict[str, str]:
    """La branche, puis sa source et son impedance prises a part."""
    tension, courant = branche_du_cinq()
    impedance = complexe(impedance_du_cinq())
    branche = complexe(puissance_complexe(tension, courant))
    source = complexe(puissance_complexe(d.CINQ_E3, courant))
    passive = complexe(puissance_complexe(impedance_du_cinq() * courant, courant))
    un, trois = complexe(d.COURANT_UN_DU_CINQ), complexe(courant)
    ecart = degres(un) - degres(trois)
    return {
        "TroisDeuxCinqETrois": nombre(float(d.CINQ_E3), 0),
        "TroisDeuxCinqRTrois": nombre(float(d.CINQ_R3), 0),
        "TroisDeuxCinqCTrois": nombre(float(d.CINQ_C3) * 1e6, 0),
        "TroisDeuxCinqFrequence": nombre(float(d.CINQ_FREQUENCE), 0),
        "TroisDeuxCinqUnReel": nombre(un.real, 1),
        "TroisDeuxCinqUnImag": nombre(un.imag, 1),
        "TroisDeuxCinqTroisReel": nombre(trois.real, 1),
        "TroisDeuxCinqTroisImag": nombre(-trois.imag, 1),
        "TroisDeuxCinqProduit": nombre(
            float(d.OMEGA_DU_CINQ * d.CINQ_R3 * d.CINQ_C3), 4
        ),
        "TroisDeuxCinqZReel": nombre(impedance.real, 2),
        "TroisDeuxCinqZImag": nombre(-impedance.imag, 2),
        "TroisDeuxCinqUReel": nombre(complexe(tension).real, 2),
        "TroisDeuxCinqUImag": nombre(-complexe(tension).imag, 2),
        "TroisDeuxCinqP": nombre(branche.real, 2),
        "TroisDeuxCinqQ": nombre(branche.imag, 2),
        "TroisDeuxCinqPSource": nombre(source.real, 0),
        "TroisDeuxCinqQSource": nombre(source.imag, 0),
        "TroisDeuxCinqPImpedance": nombre(passive.real, 2),
        "TroisDeuxCinqQImpedance": nombre(-passive.imag, 2),
        "TroisDeuxCinqModuleUn": nombre(abs(un), 4),
        "TroisDeuxCinqModuleTrois": nombre(abs(trois), 4),
        "TroisDeuxCinqPhaseUn": nombre(degres(un), 0),
        "TroisDeuxCinqPhaseTrois": nombre(degres(trois), 2),
        "TroisDeuxCinqDephasage": nombre(ecart, 2),
        "TroisDeuxCinqDephasagePrincipal": nombre(ecart - 360, 2),
    }
