"""Exercices resolus, 0.25 a 0.28 : du graphe a la transmittance.

Trois graphes de fluence canoniques, lus par la regle de Mason, et un
schema-bloc a une boucle et deux branches paralleles.
"""

from __future__ import annotations

import math
from typing import Final

import sympy as sp

from polytech.signaux.domain.fluence import transmittance_du_graphe
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.temporel import ReponsesTemporelles
from polytech.signaux.domain.transmittance import Transmittance

GRAPHES: Final = {
    "0.25": ((-20, -200), (1, 0)),
    "0.27": ((-20, -20), (1, 1)),
    "0.28": ((-1, -10, -15), (0, 0, 5)),
}
"""Les retours puis les gains directs ; 0.28 vient de son equation differentielle."""


def _schema_bloc() -> sp.Expr:
    """0.26 : `1/(p+1)`, une boucle de retour `1/p`, et `p/(p+1)` en parallele."""
    boucle = sp.cancel(1 / (1 + 1 / P))
    return sp.cancel(boucle / (P + 1) + P / (P + 1))


def transmittance(nom: str) -> sp.Expr:
    """La transmittance de l'exercice de ce numero."""
    if nom == "0.26":
        return _schema_bloc()
    retours, directs = GRAPHES[nom]
    return transmittance_du_graphe(retours=retours, directs=directs)


def limites(nom: str) -> dict[str, tuple[sp.Expr, sp.Expr]]:
    """Ce que rendent les deux theoremes, pour `h` puis pour `a`."""
    h = transmittance(nom)
    return {
        "h": (sp.limit(P * h, P, sp.oo), sp.limit(P * h, P, 0)),
        "a": (sp.limit(h, P, sp.oo), h.subs(P, 0)),
    }


def _poles(nom: str) -> list[complex]:
    denominateur = sp.fraction(sp.cancel(transmittance(nom)))[1]
    return [complex(pole) for pole in sp.Poly(denominateur, P).nroots()]


def sigma_et_omega(nom: str) -> tuple[sp.Expr, sp.Expr]:
    """`sigma` et `omega` des poles complexes `-sigma +- j omega`."""
    denominateur = sp.fraction(sp.cancel(transmittance(nom)))[1]
    pole = max(sp.roots(denominateur, P), key=sp.im)
    return -sp.re(pole), sp.im(pole)


def constantes_de_temps(nom: str) -> tuple[float, float]:
    """Les constantes de temps de deux poles reels, la plus longue d'abord."""
    lente, rapide = sorted((-1 / pole.real for pole in _poles(nom)), reverse=True)
    return lente, rapide


def poids_des_modes_de_l_impulsionnelle(nom: str) -> tuple[float, float]:
    """Les residus de `H` en ses deux poles reels, le pole lent d'abord."""
    h = sp.cancel(transmittance(nom))
    lent, rapide = sorted(sp.roots(sp.fraction(h)[1], P), key=sp.Abs)
    en_lent, en_rapide = (sp.limit((P - pole) * h, P, pole) for pole in (lent, rapide))
    return float(en_lent), float(en_rapide)


def sommet_de_l_indicielle(nom: str) -> tuple[float, float]:
    """L'instant et la hauteur du maximum de `a(t)`, pour `H = p / (p^2 + ...)`.

    L'indicielle est alors `e^(-sigma t) sin(omega t) / omega` : elle
    culmine quand `tan(omega t) = omega / sigma`.
    """
    sigma, omega = (float(valeur) for valeur in sigma_et_omega(nom))
    instant = math.atan2(omega, sigma) / omega
    return instant, math.exp(-sigma * instant) * math.sin(omega * instant) / omega


def pole_instable(nom: str) -> tuple[float, float]:
    """La partie reelle et la pulsation du pole qui fait diverger la reponse."""
    pole = max(_poles(nom), key=lambda p: (p.real, p.imag))
    return pole.real, pole.imag


def valeur_finale_verifiee(nom: str) -> sp.Expr:
    """`a(infini)`, ou une erreur si un pole interdit tout regime final."""
    reponses = ReponsesTemporelles(Transmittance(expression=transmittance(nom)))
    return reponses.valeur_finale()
