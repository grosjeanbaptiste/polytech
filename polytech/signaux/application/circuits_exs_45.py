"""Seance « Chap. 4-5 », exercices 5, 6 et 8 : trois circuits du second ordre.

Chaque transmittance vient d'un diviseur d'impedances ; ce que
l'exercice demande ensuite se lit sur son denominateur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import sympy as sp

from polytech.signaux.domain.reponse_indicielle import depassement_du_second_ordre
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T

if TYPE_CHECKING:
    from numpy.typing import NDArray

R, L, C = sp.symbols("R L C", positive=True)
DU_SIX: Final = {R: 10, C: sp.Rational(1, 10**3), L: sp.Rational(1, 10**3)}
DU_HUIT: Final = {R: 1, C: sp.Rational(1, 10**3), L: sp.Rational(1, 10**3)}
PERIODE_DU_TRAIN: Final = 0.01
"""La periode du train d'impulsions de l'exercice 6, en secondes."""


@dataclass(frozen=True, slots=True)
class ReperesDuHuit:
    """La valeur finale, les poles et le depassement relatif de l'exercice 8."""

    finale: sp.Expr
    poles: tuple[sp.Expr, ...]
    depassement: sp.Expr


def impedance_parallele() -> sp.Expr:
    """Exercice 5 : `R`, `L` et `C` en parallele, vues par la source de courant."""
    return sp.cancel(1 / (C * P + 1 / (L * P) + 1 / R))


def _second_degre() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Le denominateur de l'exercice 5, normalise : `p^2 + 2 s p + w0^2`."""
    deux, un, zero = sp.Poly(sp.fraction(impedance_parallele())[1], P).all_coeffs()
    return sp.Integer(1), un / deux, zero / deux


def resistance_limite() -> sp.Expr:
    """La resistance au-dessus de laquelle la reponse oscille."""
    _, un, zero = _second_degre()
    (limite,) = sp.solve(sp.Eq(un**2, 4 * zero), R)
    return sp.simplify(limite)


def constante_et_pulsation() -> tuple[sp.Expr, sp.Expr]:
    """`tau = 1/sigma` et la pulsation des oscillations amorties."""
    _, un, zero = _second_degre()
    sigma = un / 2
    return sp.simplify(1 / sigma), sp.sqrt(sp.simplify(zero - sigma**2))


def transmittance_du_six() -> sp.Expr:
    """`U_o/U_i` : un diviseur entre `L` et `R` ; `C` est aux bornes de la source."""
    return sp.cancel(R / (R + L * P))


def reperes_du_six() -> tuple[float, float]:
    """`h(0+) = R/L` et la constante de temps `L/R`, avec les valeurs de l'enonce."""
    return float((R / L).subs(DU_SIX)), float((L / R).subs(DU_SIX))


def periode_en_constantes_de_temps() -> float:
    """La periode du train d'impulsions, comptee en constantes de temps."""
    return PERIODE_DU_TRAIN / reperes_du_six()[1]


def reponse_au_train(instants: NDArray[np.float64]) -> NDArray[np.float64]:
    """`h_T(t)` : une reponse impulsionnelle a chaque impulsion deja passee."""
    depart, constante = reperes_du_six()
    total = np.zeros(len(instants))
    for rang in range(int(instants.max() / PERIODE_DU_TRAIN) + 1):
        ecoule = instants - rang * PERIODE_DU_TRAIN
        total += np.where(
            ecoule >= 0, depart * np.exp(-np.maximum(ecoule, 0) / constante), 0
        )
    return total


def transmittance_du_huit() -> sp.Expr:
    """`U_o/U_i` : `R`, puis `C` en parallele sur `L` et `R` en serie."""
    aval = 1 / (C * P + 1 / (R + L * P))
    return sp.cancel(aval / (R + aval) * R / (R + L * P))


def indicielle_du_huit() -> sp.Expr:
    """`u_o(t)` pour un echelon unite, avec les valeurs de l'enonce."""
    image = transmittance_du_huit().subs(DU_HUIT) / P
    brut = sp.inverse_laplace_transform(image, P, T)
    return sp.simplify(brut.subs(sp.Heaviside(T), 1))


def reperes_du_huit() -> ReperesDuHuit:
    """La valeur finale, les poles, et le depassement du second ordre."""
    chiffree = sp.cancel(transmittance_du_huit().subs(DU_HUIT))
    poles = tuple(sorted(sp.roots(sp.fraction(chiffree)[1], P), key=sp.im))
    amortissement = -sp.re(poles[0]) / sp.Abs(poles[0])
    return ReperesDuHuit(
        chiffree.subs(P, 0), poles, depassement_du_second_ordre(amortissement)
    )
