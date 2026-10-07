"""Exercices resolus, 0.22 a 0.24 : des circuits a leurs reponses.

Quatre quadripoles du premier ordre -- un element en serie, un en
sortie --, l'appariement de la seance « Chap. 4-5 », et le circuit du
second ordre de l'exemple 4.3 du syllabus.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.signaux.application import circuits_exs_45
from polytech.signaux.application.association_exs_45 import association
from polytech.signaux.domain.symboles import IMAGE as P

R, L, C = sp.symbols("R L C", positive=True)
DU_PREMIER_ORDRE: Final = {R: 100, C: sp.Rational(1, 10**9), L: sp.Rational(1, 10**3)}
"""`R = 100` ohms, `C = 1` nF, `L = 1` mH (0.22)."""
IMPEDANCES: Final = {"R": R, "L": L * P, "C": 1 / (C * P)}
DE_L_EXEMPLE: Final = {R: 1, C: sp.Rational(1, 10**3), L: sp.Rational(1, 10**3)}
"""`R = 1` ohm, `C = 1` mF, `L = 1` mH (0.24)."""


def transmittance(nom: str) -> sp.Expr:
    """Le diviseur dont `nom` donne l'element en serie puis l'element en sortie."""
    serie, sortie = (IMPEDANCES[lettre] for lettre in nom)
    return sp.cancel((sortie / (serie + sortie)).subs(DU_PREMIER_ORDRE))


def _limites(h: sp.Expr) -> dict[str, tuple[sp.Expr, sp.Expr]]:
    """Les valeurs initiale et finale de l'impulsionnelle, puis de l'indicielle."""
    return {
        "h": (sp.limit(P * h, P, sp.oo), sp.limit(P * h, P, 0)),
        "a": (sp.limit(h, P, sp.oo), h.subs(P, 0)),
    }


def limites(nom: str) -> dict[str, tuple[sp.Expr, sp.Expr]]:
    """Celles du quadripole de ce nom."""
    return _limites(transmittance(nom))


def porte_une_impulsion(nom: str) -> bool:
    """Vrai si `H` ne s'annule pas a l'infini : `h(t)` contient alors un Dirac."""
    return bool(sp.limit(transmittance(nom), P, sp.oo) != 0)


def depart_de_l_impulsionnelle(nom: str) -> sp.Expr:
    """`h(0+)`, l'impulsion de Dirac mise a part."""
    h = transmittance(nom)
    propre = sp.cancel(h - sp.limit(h, P, sp.oo))
    return sp.limit(P * propre, P, sp.oo)


def appariement() -> dict[str, tuple[str, int]]:
    """0.23 : le meme que l'exercice 4 de la seance « Chap. 4-5 »."""
    return association()


def transmittance_de_l_exemple() -> sp.Expr:
    """0.24 : `R`, puis `C` en parallele sur `L` et `R` en serie, sortie sur `R`."""
    return circuits_exs_45.transmittance_du_huit()


def limites_de_l_exemple() -> dict[str, tuple[sp.Expr, sp.Expr]]:
    """Les quatre limites de la solution, sans valeurs numeriques."""
    return _limites(transmittance_de_l_exemple())


def _sigma_et_omega() -> tuple[float, float]:
    """Les poles `-sigma +- j omega` de l'exemple, avec ses valeurs."""
    chiffree = sp.cancel(transmittance_de_l_exemple().subs(DE_L_EXEMPLE))
    pole = max(sp.roots(sp.fraction(chiffree)[1], P), key=sp.im)
    return float(-sp.re(pole)), float(sp.im(pole))


def sommet_de_l_impulsionnelle() -> tuple[float, float]:
    """L'instant et la hauteur du maximum de `h(t) = K e^(-sigma t) sin(omega t)`."""
    sigma, omega = _sigma_et_omega()
    statique = float(transmittance_de_l_exemple().subs(DE_L_EXEMPLE).subs(P, 0))
    gain = statique * (sigma**2 + omega**2) / omega
    instant = math.atan2(omega, sigma) / omega
    return instant, gain * math.exp(-sigma * instant) * math.sin(omega * instant)


def sommet_de_l_indicielle() -> float:
    """Le premier maximum de `a(t)`, atteint une demi-periode apres le depart."""
    sigma, omega = _sigma_et_omega()
    statique = float(transmittance_de_l_exemple().subs(DE_L_EXEMPLE).subs(P, 0))
    return statique * (1 + math.exp(-sigma * math.pi / omega))
