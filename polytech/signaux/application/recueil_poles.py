"""Recueil d'exercices, planches 4, 5, 6, 11 et 12 : des poles a l'allure.

La partie reelle d'un pole fixe la vitesse a laquelle la reponse
s'eteint, sa partie imaginaire la pulsation a laquelle elle oscille. Ces
planches font le trajet dans les deux sens.
"""

from __future__ import annotations

import itertools
import math
from typing import Final

import sympy as sp

from polytech.signaux.application.association_exs_45 import association
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T

SIGMA: Final = 100.0
PULSATION: Final = 2 * math.pi * 100
PHASE: Final = math.pi / 4
IMAGE_DE_LA_TENSION: Final = P / (P**2 + 2 * P + 2)
"""`U_C(p)` de la planche 5."""
CRETES: Final = ((0.8, 1.3), (1.8, 0.5), (2.8, 0.2))
"""Planche 11 : instants (s) et hauteurs (milliemes) des cretes, **lus**."""
R, C = sp.symbols("R C", positive=True)
PRODUIT_RC: Final = 10 * 1e-3
"""`R = 10` ohms et `C = 1` mF, en secondes (planche 12) ; seul le produit compte."""
CONSTANTES_POUR_S_ETEINDRE: Final = 5


def reperes_de_la_sinusoide() -> tuple[float, float, float]:
    """La periode, la constante de temps et la valeur en zero (planche 4)."""
    return 2 * math.pi / PULSATION, 1 / SIGMA, math.cos(PHASE)


def enveloppe_apres(*, periodes: int) -> float:
    """`e^(-sigma t)` au bout de ce nombre de periodes."""
    return math.exp(-SIGMA * periodes * 2 * math.pi / PULSATION)


def valeurs_extremes() -> tuple[sp.Expr, sp.Expr]:
    """`u(0+)` et `u(infini)` : les limites de `p U(p)` a l'infini et en zero."""
    produit = sp.cancel(P * IMAGE_DE_LA_TENSION)
    return sp.limit(produit, P, sp.oo), sp.limit(produit, P, 0)


def sigma_et_omega(denominateur: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`sigma` et `omega` des poles `-sigma +- j omega`."""
    pole = max(sp.roots(sp.Poly(denominateur, P)), key=sp.im)
    return -sp.re(pole), sp.im(pole)


def original_de_la_tension(instant: object) -> sp.Expr:
    """`u_C(t)`, par l'inversion formelle que la planche demande d'eviter."""
    brut = sp.inverse_laplace_transform(IMAGE_DE_LA_TENSION, P, T)
    return sp.simplify(brut.subs(sp.Heaviside(T), 1).subs(T, instant))


def appariement_des_transmittances() -> dict[str, tuple[str, int]]:
    """Planche 6 : la meme que l'exercice 4 de la seance « Chap. 4-5 »."""
    return association()


def poles_lus_sur_la_reponse() -> tuple[float, float]:
    """`sigma` et `omega` tires des cretes : decrement et periode moyens."""
    paires = list(itertools.pairwise(CRETES))
    periode = sum(suivante[0] - crete[0] for crete, suivante in paires) / len(paires)
    decrements = [math.log(crete[1] / suivante[1]) for crete, suivante in paires]
    return sum(decrements) / len(decrements) / periode, 2 * math.pi / periode


def transmittance_des_deux_cellules() -> sp.Expr:
    """Planche 12 : deux cellules `C` serie, `R` parallele, la seconde chargeant."""
    milieu, sortie = sp.symbols("V_1 V_2")
    noeuds = (
        sp.Eq((1 - milieu) * P * C, milieu / R + (milieu - sortie) * P * C),
        sp.Eq((milieu - sortie) * P * C, sortie / R),
    )
    return sp.cancel(sp.solve(noeuds, (milieu, sortie))[sortie])


def limites_de_l_impulsionnelle() -> tuple[sp.Expr, sp.Expr]:
    """`h(0+)` et `h(infini)` : les limites de `p H(p)`."""
    produit = P * transmittance_des_deux_cellules()
    return sp.limit(produit, P, sp.oo), sp.limit(produit, P, 0)


def poles_des_deux_cellules() -> tuple[float, float]:
    """Les deux poles, en rad/s, du plus lent au plus rapide."""
    denominateur = sp.fraction(transmittance_des_deux_cellules())[1]
    chiffre = denominateur.subs({R: PRODUIT_RC, C: 1})
    lent, rapide = sorted(
        (float(pole) for pole in sp.Poly(chiffre, P).nroots()), key=abs
    )
    return lent, rapide


def extinction() -> tuple[float, float]:
    """La plus grande constante de temps, et cinq fois celle-ci."""
    constante = -1 / poles_des_deux_cellules()[0]
    return constante, CONSTANTES_POUR_S_ETEINDRE * constante
