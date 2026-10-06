"""Les quatre methodes de dimensionnement du Sallen-Key passe-bas.

Chapitre 3, partie II, planches 24 a 27. Trois equations pour six
composants : chaque methode fixe ce qui reste libre, et paie ce choix.

1. *Elements egaux* : `C_1 = C_2 = 1`, `R_1 = R_2 = 1 / omega_0`, et le gain
   `mu = 3 - 1/Q` -- la seule ou le gain se regle, mais lie a `Q`.
2. *Capacites egales*, `R_a = R_b` : `mu = 2`, `R_1 / R_2 = Q^2`.
3. *Sensibilite moderee* : `C_1 = racine(3) Q`, `C_2 = 1`, `mu = 4/3`.
4. *Sensibilite minimale* : `R_1 = R_2 = 1`, `mu = 1` (un suiveur), et
   `C_1 / C_2 = 4 Q^2`.

Les valeurs sont *normalisees* -- des ohms et des farads de l'ordre de
l'unite -- et se ramenent a des composants par
`SallenKeyPasseBas.denormalisee`.
"""

from __future__ import annotations

import enum
import math

import sympy as sp

from polytech.electronique.domain.sallen_key_a_gain import SallenKeyPasseBas


class Methode(enum.Enum):
    """Les quatre « designs » des planches 24 a 27, dans leur ordre."""

    ELEMENTS_EGAUX = 1
    CAPACITES_EGALES = 2
    SENSIBILITE_MODEREE = 3
    SENSIBILITE_MINIMALE = 4


_GAINS_FIXES = {
    Methode.CAPACITES_EGALES: sp.Integer(2),
    Methode.SENSIBILITE_MODEREE: sp.Rational(4, 3),
    Methode.SENSIBILITE_MINIMALE: sp.Integer(1),
}


def gain_de_la_methode(methode: Methode, qualite: sp.Expr) -> sp.Expr:
    """Le gain continu que la methode impose : `3 - 1/Q`, 2, 4/3 ou 1."""
    if methode is Methode.ELEMENTS_EGAUX:
        return 3 - 1 / sp.sympify(qualite)
    return _GAINS_FIXES[methode]


def cellule_normalisee(
    methode: Methode, qualite: sp.Expr, pulsation: sp.Expr
) -> SallenKeyPasseBas:
    """La cellule que la methode donne pour `Q` et `omega_0`."""
    q, w = sp.sympify(qualite), sp.sympify(pulsation)
    un, gain = sp.Integer(1), gain_de_la_methode(methode, q)
    if methode is Methode.ELEMENTS_EGAUX:
        return SallenKeyPasseBas(1 / w, 1 / w, un, un, gain)
    if methode is Methode.CAPACITES_EGALES:
        return SallenKeyPasseBas(q / w, 1 / (q * w), un, un, gain)
    if methode is Methode.SENSIBILITE_MODEREE:
        racine = sp.sqrt(3)
        return SallenKeyPasseBas(1 / (w * q), 1 / (racine * w), racine * q, un, gain)
    return SallenKeyPasseBas(un, un, 2 * q / w, 1 / (2 * q * w), gain)


def qualite_des_elements_egaux(gain: float) -> float:
    """Le `Q` que la methode 1 impose a ce gain : `mu = 3 - 1/Q` renverse.

    C'est la seule methode dont le gain se regle, et il n'est pas libre :
    a 3 la cellule oscille, au-dela elle diverge.
    """
    plafond = 3.0
    if gain >= plafond:
        msg = f"gain de trois ou plus, la cellule n'est plus un filtre : {gain}"
        raise ValueError(msg)
    return 1 / (plafond - gain)


def methodes_a_gain_fixe(gain: float) -> tuple[Methode, ...]:
    """Les methodes 2 a 4 dont le gain impose est celui-ci ; `Q` y reste libre."""
    return tuple(m for m, fixe in _GAINS_FIXES.items() if math.isclose(gain, fixe))


def resistances_de_gain(gain: float, resistance_vue: float) -> tuple[float, float]:
    """`(R_a, R_b)` : `1 + R_b / R_a = mu` et `R_a // R_b = resistance_vue`.

    Planche 30 : pour que les courants de polarisation des deux entrees
    produisent le meme decalage, l'entree inverseuse doit voir la meme
    resistance que l'autre, `R_1 + R_2`. Comme `R_a // R_b = R_b / mu`,
    `R_b = mu (R_1 + R_2)` et `R_a = R_b / (mu - 1)`.
    """
    if gain <= 1:
        msg = f"gain inferieur ou egal a un, sans pont de gain : {gain}"
        raise ValueError(msg)
    if resistance_vue <= 0:
        msg = f"resistance vue nulle ou negative : {resistance_vue}"
        raise ValueError(msg)
    retour = gain * resistance_vue
    return retour / (gain - 1), retour
