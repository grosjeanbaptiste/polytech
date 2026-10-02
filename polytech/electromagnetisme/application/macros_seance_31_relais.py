"""Macros de la seance 3.1, exercice 1 : le relais.

Le seul numerique de la seance. Les valeurs sont celles du manuscrit --
0,226 ampere, 21,24 par seconde -- recalculees depuis des rationnels
exacts, plus trois que le manuscrit ne donne pas : la constante de
temps, l'instant d'enclenchement a une decimale de plus, et le courant
atteint a l'instant qu'annoncent le recueil et le manuscrit.

Le prefixe est `TroisUnUn`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.enonces_seance_31_relais import (
    BOBINE_DU_RELAIS,
    CONSTANTE_DE_TEMPS,
    RESISTANCE_TOTALE,
    circuit_du_relais,
    courant_final_du_relais,
    instant_d_enclenchement,
)
from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.application.symboles_seance_31 import (
    FEM_DU_RELAIS,
    INDUCTANCE_DE_LA_BOBINE,
    RESISTANCE_DE_LA_BOBINE,
    RESISTANCE_DE_LIGNE,
    RESISTANCE_INTERNE,
    SEUIL_DU_RELAIS,
)
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import original

INSTANT_ANNONCE = sp.Rational(11, 1000)
"""Les onze millisecondes que donnent le recueil et le manuscrit."""


def _macros_des_donnees() -> dict[str, str]:
    """Les six donnees du recueil."""
    return {
        "TroisUnUnFem": nombre(float(FEM_DU_RELAIS), 0),
        "TroisUnUnResistanceInterne": nombre(float(RESISTANCE_INTERNE), 1),
        "TroisUnUnResistanceDeLigne": nombre(float(RESISTANCE_DE_LIGNE), 0),
        "TroisUnUnResistanceDeLaBobine": nombre(float(RESISTANCE_DE_LA_BOBINE), 0),
        "TroisUnUnInductance": nombre(float(INDUCTANCE_DE_LA_BOBINE), 1),
        "TroisUnUnSeuil": nombre(float(SEUIL_DU_RELAIS), 2),
    }


def _macros_du_calcul() -> dict[str, str]:
    """La valeur finale, la constante de temps, l'instant, et le controle."""
    final = courant_final_du_relais()
    courant = original(circuit_du_relais().courant(BOBINE_DU_RELAIS))
    return {
        "TroisUnUnResistanceTotale": nombre(float(RESISTANCE_TOTALE), 1),
        "TroisUnUnCourantFinal": nombre(float(final), 3),
        "TroisUnUnCoefficient": nombre(float(-INDUCTANCE_DE_LA_BOBINE * final), 3),
        "TroisUnUnInverseDeTau": nombre(float(1 / CONSTANTE_DE_TEMPS), 2),
        "TroisUnUnTau": nombre(float(CONSTANTE_DE_TEMPS) * 1e3, 1),
        "TroisUnUnInstant": nombre(float(instant_d_enclenchement()) * 1e3, 1),
        "TroisUnUnCourantAOnze": nombre(
            float(courant.subs(TEMPS, INSTANT_ANNONCE)),
            3,
        ),
    }


def macros_du_relais() -> dict[str, str]:
    """Les donnees, puis le calcul."""
    return {**_macros_des_donnees(), **_macros_du_calcul()}
