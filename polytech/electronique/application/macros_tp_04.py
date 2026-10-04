"""Macros du TP 4 d'electronique : matrices nodales, gains, pole et zero.

Les matrices sont celles que la bibliotheque *assemble* ; les grands
nombres sont publies en notation `mantisseeexposant`, que siunitx lit.

Le prefixe est `TpQuatre`.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electronique.application import circuits_tp_04 as ct
from polytech.electronique.application import deductions_tp_04 as dd
from polytech.electronique.application import nombres_tp_04 as nb
from polytech.electronique.application.formatage import nombre
from polytech.electronique.application.symboles_tp_04 import VS


def _latex(expression: sp.Basic) -> str:
    return str(sp.latex(expression, imaginary_unit="j"))


def _scientifique(valeur: float, decimales: int) -> str:
    """`8.33e12` : la mantisse arrondie, l'exposant entier."""
    exposant = math.floor(math.log10(abs(valeur)))
    return f"{nombre(valeur / 10**exposant, decimales)}e{exposant}"


def _macros_des_systemes() -> dict[str, str]:
    macros: dict[str, str] = {}
    for nom, systeme in (
        ("Un", ct.systeme_du_circuit_1),
        ("Deux", ct.systeme_du_circuit_2),
        ("Trois", ct.systeme_du_circuit_3),
        ("Dephaseur", ct.systeme_du_dephaseur),
        ("SourceCommune", ct.systeme_de_la_source_commune),
        ("Cascode", ct.systeme_de_la_cascode),
    ):
        matrice, membre = systeme()
        macros[f"TpQuatreMatrice{nom}"] = _latex(matrice)
        macros[f"TpQuatreMembre{nom}"] = _latex(membre)
    macros["TpQuatreVDeuxUn"] = _latex(sp.factor(ct.solution_du_circuit_1()[1]))
    return macros


def _macros_symboliques() -> dict[str, str]:
    v3, v4 = dd.tensions_de_la_cascode()
    return {
        "TpQuatreHDephaseur": _latex(sp.factor(dd.transfert_du_dephaseur())),
        "TpQuatrePulsation": _latex(dd.pulsation_d_inversion()),
        "TpQuatreInversion": _latex(dd.gain_a_l_inversion()),
        "TpQuatreHSourceCommune": _latex(dd.transfert_de_la_source_commune()),
        "TpQuatreGainSansCapacite": _latex(dd.gain_sans_capacite()),
        "TpQuatreGainSansDegenerescence": _latex(dd.gain_sans_degenerescence()),
        "TpQuatreVTroisCascode": _latex(sp.factor(v3 / VS)),
        "TpQuatreVQuatreCascode": _latex(sp.factor(v4 / VS)),
        "TpQuatreCourantCharge": _latex(sp.factor(dd.courant_dans_la_charge())),
        "TpQuatreGainAVide": _latex(sp.factor(dd.gain_de_la_cascode_a_vide())),
    }


def _macros_numeriques() -> dict[str, str]:
    sc, cas = nb.source_commune_numerique(), nb.cascode_numerique()
    deux_pi = 2 * math.pi
    return {
        "TpQuatreGainContinu": nombre(sc.gain_en_continu, 2),
        "TpQuatreGainInfini": nombre(sc.gain_a_l_infini, 4),
        "TpQuatreGainInfiniApproche": nombre(sc.gain_a_l_infini_approche, 4),
        "TpQuatreAv": nombre(sc.gain_av, 0),
        "TpQuatreMillerFemto": nombre(sc.capacite_miller * 1e15, 0),
        "TpQuatreZero": _scientifique(sc.zero, 3),
        "TpQuatrePole": _scientifique(sc.pole, 3),
        "TpQuatrePoleApproche": _scientifique(sc.pole_approche, 3),
        "TpQuatreFrequenceZero": _scientifique(sc.zero / deux_pi, 3),
        "TpQuatreFrequencePole": _scientifique(-sc.pole / deux_pi, 3),
        "TpQuatreFrequencePoleApproche": _scientifique(-sc.pole_approche / deux_pi, 3),
        "TpQuatreEcartPole": nombre(100 * (sc.pole_approche / sc.pole - 1), 2),
        "TpQuatreGainInterne": nombre(cas.gain_interne, 4),
        "TpQuatreGainAVideNumerique": nombre(cas.gain_a_vide, 0),
        "TpQuatreFacteurMiller": nombre(cas.facteur_miller, 4),
    }


def macros_du_tp_quatre() -> dict[str, str]:
    """Les systemes, les expressions, les nombres."""
    return {**_macros_des_systemes(), **_macros_symboliques(), **_macros_numeriques()}
