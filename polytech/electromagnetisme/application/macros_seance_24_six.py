"""Macros de la seance 2.4, exercice 6 : le seul numerique.

Le generateur de Thevenin, puis la puissance dans la charge par **deux**
routes : celle du manuscrit, qui passe par le generateur, et le reseau
complet ou la charge et sa source sont branchees. La seconde n'emploie
aucun theoreme.

S'y ajoute le courant de la branche `E_5`-`R_5` : elle ne change rien au
generateur, mais elle n'est pas inerte pour autant -- un courant y
circule, que le manuscrit ne calcule pas.

Le prefixe est `DeuxQuatreSix`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.electromagnetisme.application.enonces_seance_24 import (
    SOURCE_CINQ,
    reseau_charge_du_six,
    reseau_du_six,
)
from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.application.symboles_seance_24 import (
    CHARGE_PRIME,
    COURANT_PRIME,
    DONNEES_DU_SIX,
    R1,
    R3,
    R4,
    R5,
    TENSION_1,
    TENSION_5,
)
from polytech.electromagnetisme.domain.circuits.equivalents import (
    resistance_vue_des_bornes,
    tension_a_vide,
)

if TYPE_CHECKING:
    import sympy as sp


def _valeur(expression: sp.Expr, decimales: int) -> str:
    """L'expression evaluee sur les donnees de l'enonce."""
    return nombre(float(expression.subs(DONNEES_DU_SIX)), decimales)


def _macros_des_donnees() -> dict[str, str]:
    """Les huit donnees du recueil, dont deux ne servent a rien."""
    return {
        "DeuxQuatreSixRUn": _valeur(R1, 0),
        "DeuxQuatreSixRTrois": _valeur(R3, 0),
        "DeuxQuatreSixRQuatre": _valeur(R4, 0),
        "DeuxQuatreSixRCinq": _valeur(R5, 0),
        "DeuxQuatreSixEUn": _valeur(TENSION_1, 0),
        "DeuxQuatreSixECinq": _valeur(TENSION_5, 0),
        "DeuxQuatreSixCharge": _valeur(CHARGE_PRIME, 1),
        "DeuxQuatreSixCourantPrime": _valeur(COURANT_PRIME, 0),
    }


def _macros_du_calcul() -> dict[str, str]:
    """Le generateur, le courant et la puissance, par les deux routes."""
    nu = reseau_du_six()
    a_vide = tension_a_vide(nu, "A", "B")
    resistance = resistance_vue_des_bornes(nu, "A", "B")
    par_thevenin = (a_vide - resistance * COURANT_PRIME) / (CHARGE_PRIME + resistance)
    potentiels = reseau_charge_du_six().potentiels()
    aux_bornes = potentiels["A"] - potentiels["B"]
    inerte = -nu.courant_debite(SOURCE_CINQ)
    return {
        "DeuxQuatreSixResistance": _valeur(resistance, 1),
        "DeuxQuatreSixTension": _valeur(a_vide, 0),
        "DeuxQuatreSixCourantDeMaille": _valeur(TENSION_1 / (R1 + R3 + R4), 0),
        "DeuxQuatreSixCourant": _valeur(par_thevenin, 1),
        "DeuxQuatreSixPuissance": _valeur(CHARGE_PRIME * par_thevenin**2, 3),
        "DeuxQuatreSixTensionAuxBornes": _valeur(aux_bornes, 2),
        "DeuxQuatreSixPuissanceDirecte": _valeur(aux_bornes**2 / CHARGE_PRIME, 3),
        "DeuxQuatreSixCourantDeLaBrancheCinq": _valeur(inerte, 0),
    }


def macros_du_six() -> dict[str, str]:
    """Les donnees, puis le calcul par ses deux routes."""
    return {**_macros_des_donnees(), **_macros_du_calcul()}
