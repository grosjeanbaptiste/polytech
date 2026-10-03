"""Macros du laboratoire III : deux Thevenin, un R-L, un filtre.

Le montage simple est publie avec la valeur que l'officiel mesure sous
QUCS (le courant sous un volt), pour que le lecteur retrouve la meme
lecture. Le montage a dix resistances, que l'officiel laisse en blanc,
est publie avec ses deux seconds chemins : les etapes de Kennelly et
les trois parts de la superposition.

Le prefixe est `LaboTrois`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.enonces_labo_03 import (
    circuit_a_dix_resistances,
    circuit_simple,
    etapes_de_kennelly,
    frequence_de_coupure,
    module_du_filtre,
    resistance_par_kennelly,
    tension_de_bobine,
)
from polytech.electromagnetisme.application.formatage import expression_latex, nombre
from polytech.electromagnetisme.domain.circuits.equivalents import (
    courant_de_court_circuit,
    resistance_vue_des_bornes,
    tension_a_vide,
)

R_INDICIELLE, L_INDICIELLE = 10_000, 1
R_FILTRE, L_FILTRE = 2200, sp.Rational(484, 10000)
FREQUENCE_MAXIMALE = 35_000
"""La fin du balayage de l'officiel, 35 kHz."""
NOMS_DES_ETAPES = (
    "Gauche",
    "VersTDeux",
    "EtoileT",
    "EtoileM",
    "EtoileN",
    "ParM",
    "ParN",
    "AvecRTrois",
)


def _macros_du_simple() -> dict[str, str]:
    """La tension, la resistance, et le courant que QUCS lit sous un volt."""
    reseau = circuit_simple()
    resistance = resistance_vue_des_bornes(reseau, "a", "m")
    return {
        "LaboTroisSimpleE": nombre(float(tension_a_vide(reseau, "a", "m")), 0),
        "LaboTroisSimpleR": expression_latex(resistance),
        "LaboTroisSimpleRDecimal": nombre(float(resistance), 3),
        "LaboTroisSimpleCourant": nombre(float(1 / resistance), 1),
    }


def _macros_des_dix() -> dict[str, str]:
    """Thevenin, Norton, la superposition et les etapes de Kennelly."""
    reseau = circuit_a_dix_resistances()
    tension = tension_a_vide(reseau, "a", "b")
    parts = [
        float(tension_a_vide(circuit_a_dix_resistances(*s), "a", "b"))
        for s in ((10, 0, 0), (0, 5, 0), (0, 0, 1))
    ]
    etapes = {
        f"LaboTroisKennelly{nom}": nombre(float(valeur), 4)
        for nom, valeur in zip(NOMS_DES_ETAPES, etapes_de_kennelly(), strict=True)
    }
    return {
        "LaboTroisDixE": expression_latex(tension),
        "LaboTroisDixEDecimal": nombre(float(tension), 4),
        "LaboTroisDixR": expression_latex(resistance_par_kennelly()),
        "LaboTroisDixRDecimal": nombre(float(resistance_par_kennelly()), 4),
        "LaboTroisDixCourant": nombre(
            float(courant_de_court_circuit(reseau, "a", "b")), 4
        ),
        "LaboTroisPartEUn": nombre(parts[0], 4),
        "LaboTroisPartEDeux": nombre(parts[1], 4),
        "LaboTroisPartI": nombre(parts[2], 4),
        **etapes,
    }


def _macros_du_rl() -> dict[str, str]:
    """La reponse indicielle, sa constante de temps, trois points de lecture."""
    v0, r, inductance = sp.symbols("v_0 R L", positive=True)
    tau = L_INDICIELLE / R_INDICIELLE
    return {
        "LaboTroisTension": expression_latex(tension_de_bobine(v0, r, inductance)),
        "LaboTroisTau": nombre(tau * 1e6, 0),
        "LaboTroisAUnTau": nombre(float(sp.exp(-1)), 3),
        "LaboTroisATroisTau": nombre(float(sp.exp(-3)), 3),
        "LaboTroisACinqTau": nombre(float(sp.exp(-5)), 4),
    }


def _macros_du_filtre() -> dict[str, str]:
    """La frequence de coupure, le module qu'elle donne, et la fin du balayage."""
    coupure = frequence_de_coupure(R_FILTRE, L_FILTRE)
    return {
        "LaboTroisCoupure": nombre(float(coupure), 0),
        "LaboTroisCoupureKilo": nombre(float(coupure) / 1e3, 3),
        "LaboTroisModuleCoupure": nombre(
            float(module_du_filtre(R_FILTRE, L_FILTRE, coupure)), 4
        ),
        "LaboTroisModuleFin": nombre(
            float(module_du_filtre(R_FILTRE, L_FILTRE, FREQUENCE_MAXIMALE)), 3
        ),
    }


def macros_du_labo_trois() -> dict[str, str]:
    """Les deux parties du laboratoire."""
    return {
        **_macros_du_simple(),
        **_macros_des_dix(),
        **_macros_du_rl(),
        **_macros_du_filtre(),
    }
