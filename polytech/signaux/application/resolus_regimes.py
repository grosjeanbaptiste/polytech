"""Exercices resolus, chapitre 5 (0.29 a 0.35) : regimes et transitoires.

Les exercices 0.29 et 0.31 a 0.34 sont ceux de la seance « Chap. 4-5 » et
du recueil, dont les calculs sont repris. Restent l'enclenchement d'un
circuit `R`-`L` sur une tension sinusoidale (0.30) et une entree
periodique dans un passe-bas (0.35).
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.signaux.application import (
    amortie_exs_45,
    circuits_exs_45,
    resolus_quadripoles,
    systemes_exs_45,
)
from polytech.signaux.domain.raies import raie_de_sortie
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T
from polytech.signaux.domain.transformees import transformee_de_laplace

R, L, C = sp.symbols("R L C", positive=True)
E, W = sp.symbols("E omega", positive=True)
PHI: Final = sp.Symbol("phi", real=True)
PULSATION_FONDAMENTALE: Final = 2 * sp.pi * 10
"""`w0`, pour l'entree periodique de 10 Hz de l'exercice 0.35."""

duree_et_periode_de_la_tension = amortie_exs_45.duree_et_periode
tension_de_la_capacite = amortie_exs_45.original
resistance_limite = circuits_exs_45.resistance_limite
transmittance_sans_le_condensateur = circuits_exs_45.transmittance_du_six
reperes_de_l_impulsionnelle = circuits_exs_45.reperes_du_six
periode_du_train_en_constantes = circuits_exs_45.periode_en_constantes_de_temps
alpha_limite = systemes_exs_45.alpha_limite
sommet_de_l_indicielle = resolus_quadripoles.sommet_de_l_indicielle


def limites_de_l_indicielle() -> tuple[sp.Expr, sp.Expr]:
    """0.34 : `a(0+)` et `a(infini)` du circuit de l'exemple 4.3."""
    return resolus_quadripoles.limites_de_l_exemple()["a"]


def admittance() -> tuple[sp.Expr, sp.Expr]:
    """0.30 : le module `Y` et l'argument `theta` de `1/(R + j w L)`."""
    return 1 / sp.sqrt(R**2 + (W * L) ** 2), -sp.atan(W * L / R)


def amplitude_du_transitoire(phase: sp.Expr = PHI) -> sp.Expr:
    """`Y E racine(2) cos(phi + theta)` : ce que le transitoire vaut en zero."""
    module, argument = admittance()
    return sp.simplify(module * E * sp.sqrt(2) * sp.cos(phase + argument))


def courant() -> sp.Expr:
    """`i(t)` de la solution : le regime, moins un transitoire qui l'annule en zero."""
    module, argument = admittance()
    regime = module * E * sp.sqrt(2) * sp.cos(W * T + PHI + argument)
    return regime - amplitude_du_transitoire() * sp.exp(-T * R / L)


def residu_de_la_maille() -> sp.Expr:
    """`L i' + R i - e(t)`, qui doit etre nul a tout instant."""
    tension = E * sp.sqrt(2) * sp.cos(W * T + PHI)
    maille = L * courant().diff(T) + R * courant() - tension
    return sp.simplify(sp.expand_trig(maille))


def phase_du_pire_cas() -> sp.Expr:
    """Le `phi` qui rend le transitoire maximal : `phi + theta = 0`."""
    return -admittance()[1]


def crete_relative(*, constante_en_periodes: float) -> float:
    """La premiere crete du pire cas, rapportee a l'amplitude de regime.

    Avec `phi + theta = 0`, `i` est proportionnel a `cos(w t) - e^(-t/tau)` :
    sa premiere crete, en `w t = pi`, vaut `1 + e^(-T/(2 tau))`.
    """
    return 1 + math.exp(-1 / (2 * constante_en_periodes))


def reponse_en_frequence(pulsation: sp.Expr) -> sp.Expr:
    """0.35 : la transformee de Laplace de `h(t) = e^(-t)`, prise en `p = j w`."""
    return transformee_de_laplace(sp.exp(-T)).subs(P, sp.I * pulsation)


def coefficient_de_sortie(rang: sp.Expr) -> sp.Expr:
    """`Y_k = X_k H(j k w0)`, avec `X_k = 1/(1 + jk)`."""
    entree = 1 / (1 + sp.I * rang)
    passe_bas = transformee_de_laplace(sp.exp(-T))
    return raie_de_sortie(entree, passe_bas, rang, PULSATION_FONDAMENTALE)


def attenuation_du_fondamental() -> float:
    """Le facteur par lequel le passe-bas divise la raie de rang un."""
    return float(1 / sp.Abs(reponse_en_frequence(PULSATION_FONDAMENTALE)))
