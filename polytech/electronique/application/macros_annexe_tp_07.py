"""Macros de l'annexe du TP 7 d'electronique. Prefixe `TpSeptBis`."""

from __future__ import annotations

import math

import numpy as np
import sympy as sp

from polytech.electronique.application import annexe_tp_07 as an
from polytech.electronique.application import annexe_tp_07_actifs as ac
from polytech.electronique.application.formatage import coordonnees, nombre
from polytech.electronique.domain.methodes_sallen_key import Methode

_NOMS = ("Un", "Deux", "Trois", "Quatre", "Cinq")
_COUPURE, _CENTRE, _LARGEUR = 1000.0, 1000.0, 200.0
"""Les pulsations des trois filtres traces, en rad/s."""
_QUALITE = 1 / math.sqrt(2)
"""Le facteur de qualite d'un Butterworth d'ordre 2, pour chiffrer les methodes."""


def _ligne(valeurs: tuple[float, ...]) -> str:
    """Les cases d'une ligne de table, completee de cases vides jusqu'a six."""
    cases = [f"\\num{{{nombre(v, 4)}}}" for v in valeurs]
    return " & ".join(cases + [""] * (6 - len(cases)))


def _tables() -> dict[str, str]:
    macros = {}
    for ordre, nom in enumerate(_NOMS, start=1):
        butterworth = (*an.elements_de_butterworth(ordre), 1.0)
        macros[f"TpSeptBisButterworth{nom}"] = _ligne(butterworth)
        macros[f"TpSeptBisChebyshev{nom}"] = _ligne(an.elements_de_chebyshev(ordre))
    exact = an.premier_element_exact()
    ecart = an.elements_de_chebyshev(1)[0] - exact
    return macros | {
        "TpSeptBisPremierExact": nombre(exact, 5),
        "TpSeptBisEcart": nombre(ecart * 1e5, 1),
        "TpSeptBisChargePaire": nombre(an.elements_de_chebyshev(2)[-1], 4),
    }


def _gain(transfert: sp.Expr, pulsations: np.ndarray) -> str:
    """Le module en decibels, rapporte au gain de 1/2 des terminaisons egales."""
    reponse = sp.lambdify(an.S, transfert, "numpy")(1j * pulsations)
    return coordonnees(pulsations, 20 * np.log10(np.abs(reponse) * 2), 3)


def _courbes() -> dict[str, str]:
    pulsations = np.logspace(1.5, 4.5, 241)
    return {
        "TpSeptBisPasseBas": _gain(an.vers_passe_bas(_COUPURE), pulsations),
        "TpSeptBisPasseHaut": _gain(an.vers_passe_haut(_COUPURE), pulsations),
        "TpSeptBisPasseBande": _gain(
            an.vers_passe_bande(_CENTRE, _LARGEUR), pulsations
        ),
    }


def _methodes() -> dict[str, str]:
    macros = {}
    valeurs = {ac.Q: _QUALITE, ac.W0: 1}
    for methode, nom in zip(Methode, _NOMS, strict=False):
        cellule = ac.composants(methode)
        champs = (cellule.r1, cellule.r2, cellule.c1, cellule.c2, cellule.gain)
        cases = (f"\\num{{{nombre(float(c.subs(valeurs)), 3)}}}" for c in champs)
        macros[f"TpSeptBisMethode{nom}"] = " & ".join(cases)
    ra, rb = ac.pont_de_gain(gain=2.0, r1=4000.0, r2=6000.0)
    return macros | {
        "TpSeptBisPontA": nombre(ra / 1e3, 1),
        "TpSeptBisPontB": nombre(rb / 1e3, 1),
    }


def macros_de_l_annexe() -> dict[str, str]:
    """Les macros des quatre sections de l'annexe."""
    return {**_tables(), **_courbes(), **_methodes()}
