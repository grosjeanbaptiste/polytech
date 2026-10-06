r"""Macros des exercices 2 de mathematique, II.6 et II.7 : reponses et courbes."""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import courbes_exs_02 as co
from polytech.maths.application import laplace_exs_02 as la
from polytech.maths.application import reponses_exs_02 as re
from polytech.maths.application.formatage import coordonnees, formule
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

W = la.PULSATION


def _reponses() -> dict[str, str]:
    pendant, _ = re.au_creneau()
    indicielle, impulsionnelle, sinusoidal = re.courants()
    amplitude, etabli = re.regime_etabli()
    return {
        "MathDeuxIndicielle": formule(sp.expand(re.indicielle())),
        "MathDeuxCreneauPendant": formule(sp.expand(pendant)),
        "MathDeuxImpulsionnelle": formule(sp.expand(re.impulsionnelle())),
        "MathDeuxAdmittance": formule(sp.factor(re.admittance())),
        "MathDeuxCourantEchelon": formule(sp.expand(indicielle)),
        "MathDeuxCourantImpulsion": formule(sp.expand(impulsionnelle)),
        "MathDeuxCourantSinus": formule(sp.expand(sinusoidal)),
        "MathDeuxResidu": formule(re.residu_en_j()),
        "MathDeuxAmplitude": formule(amplitude),
        "MathDeuxEtabli": formule(etabli),
        "MathDeuxCourantGeneral": formule(par_regime(re.courant_sinusoidal(W))),
        "MathDeuxPulsationDeResonance": formule(re.resonance_du_circuit()[0]),
        "MathDeuxAmplitudeMaximale": formule(re.resonance_du_circuit()[1]),
        "MathDeuxAmplitudeGenerale": formule(
            sp.simplify(sp.Abs(re.admittance().subs(P, sp.I * W)))
        ),
    }


def par_regime(courant: sp.Expr) -> sp.Expr:
    """Le courant, range par regime : chaque exponentielle, le sinus, le cosinus."""
    regimes = [sp.exp(-T), sp.exp(-2 * T), sp.sin(W * T), sp.cos(W * T)]
    return sp.collect(sp.expand(courant), regimes, sp.factor)


def _courbes() -> dict[str, str]:
    temps, creneaux, impulsionnelle = co.courbes_des_creneaux()
    macros = {"MathDeuxCourbeImpulsionnelle": coordonnees(temps, impulsionnelle, 4)}
    for lettre, reponse in zip("ABC", creneaux.values(), strict=True):
        macros[f"MathDeuxCourbeCreneau{lettre}"] = coordonnees(temps, reponse, 4)
    instants, courant, etabli = co.courbes_du_courant()
    macros["MathDeuxCourbeCourant"] = coordonnees(instants, courant, 4)
    macros["MathDeuxCourbeEtabli"] = coordonnees(instants, etabli, 4)
    return macros


def macros_des_reponses() -> dict[str, str]:
    """Les formules de II.6 et II.7, et les courbes de la seance II."""
    return {**_reponses(), **_courbes()}
