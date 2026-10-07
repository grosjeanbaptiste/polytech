"""Recueil d'exercices, planches 2, 3 et 17 : spectres et puissance.

Une salve de cosinus et son spectre a deux lobes ; une rampe isolee, puis
repetee, dont le spectre devient un spectre de raies ; la puissance d'une
salve de sinusoide repetee toutes les `T` secondes.
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.signaux.domain.frequence import FREQUENCE as F
from polytech.signaux.domain.raies import raie
from polytech.signaux.domain.symboles import TEMPS as T
from polytech.signaux.domain.transformees import puissance_moyenne_d_un_signal

if TYPE_CHECKING:
    from collections.abc import Sequence

F0: Final = sp.Symbol("f_0", positive=True)
PERIODES: Final = 8
"""Le nombre d'oscillations de la salve, compte sur la planche 2."""
DUREE: Final = PERIODES / F0
"""`tau`, la duree de la salve : un nombre entier de periodes."""
REPETITION: Final = sp.Integer(10)
"""`T0`, en secondes, de l'onde periodique de la planche 3."""
AMPLITUDE: Final = 2
SALVE: Final = sp.Rational(1, 100)
"""Dix periodes a 1000 Hz, en secondes (planche 17)."""
_INSTANT: Final = sp.Symbol("u", real=True)


def _spectre_de_la_salve() -> sp.Expr:
    """La transformee de Fourier de `cos(2 pi f0 t)` sur `[-tau/2, tau/2]`."""
    noyau = sp.cos(2 * sp.pi * F0 * _INSTANT) * sp.exp(-2 * sp.pi * sp.I * F * _INSTANT)
    return sp.integrate(noyau, (_INSTANT, -DUREE / 2, DUREE / 2), conds="none")


def spectre_de_la_salve(frequence: object) -> sp.Expr:
    """Le spectre de la salve en cette frequence, prolonge par continuite."""
    return sp.simplify(sp.limit(_spectre_de_la_salve(), F, sp.sympify(frequence)))


def spectre_normalise(rapports: Sequence[float]) -> list[float]:
    """`|E(f)| / (tau/2)` pour ces valeurs de `f / f0`."""
    reduit = sp.simplify((_spectre_de_la_salve() / (DUREE / 2)).subs(F0, 1))
    rapide = sp.lambdify(F, reduit, "cmath")
    valeurs = []
    for rapport in rapports:
        try:
            valeurs.append(abs(complex(rapide(float(rapport)))))
        except ZeroDivisionError:  # en 0 et en +-f0 l'expression s'ecrit 0/0
            valeurs.append(abs(complex(sp.limit(reduit, F, sp.nsimplify(rapport)))))
    return valeurs


def hauteur_du_lobe() -> sp.Expr:
    """La valeur du spectre en `f = f0`."""
    return spectre_de_la_salve(F0)


def largeur_du_lobe() -> sp.Expr:
    """L'ecart entre les deux zeros qui encadrent `f0`."""
    return sp.Integer(2) / DUREE


def _image_de_la_rampe() -> sp.Expr:
    """La transformee de `x(t) = t` sur `[0, 1]`, fonction de la frequence."""
    noyau = _INSTANT * sp.exp(-2 * sp.pi * sp.I * F * _INSTANT)
    return sp.integrate(noyau, (_INSTANT, 0, 1))


def image_de_la_rampe(frequence: object) -> sp.Expr:
    """Cette transformee en une frequence, prolongee par continuite."""
    return sp.simplify(sp.limit(_image_de_la_rampe(), F, sp.sympify(frequence)))


def modules_de_la_rampe(frequences: Sequence[float]) -> list[float]:
    """`|X(f)|` pour ces frequences, en hertz."""
    image = _image_de_la_rampe()
    rapide = sp.lambdify(F, image, "cmath")
    valeurs = []
    for frequence in frequences:
        try:
            valeurs.append(abs(complex(rapide(float(frequence)))))
        except ZeroDivisionError:  # a frequence nulle l'expression s'ecrit 0/0
            valeurs.append(abs(complex(sp.limit(image, F, 0))))
    return valeurs


def niveau_en_decibels(frequence: object) -> float:
    """`20 log |X(f)|`."""
    return 20 * math.log10(abs(complex(image_de_la_rampe(frequence))))


def phase_en_degres(frequence: object) -> float:
    """L'argument de `X(f)`."""
    return math.degrees(cmath.phase(complex(image_de_la_rampe(frequence))))


def ecart_des_raies() -> sp.Expr:
    """`1/T0`, en hertz."""
    return 1 / REPETITION


def raie_de_l_onde_periodique(rang: int) -> sp.Expr:
    """`X(k/T0)/T0` : la raie de ce rang."""
    return raie(_image_de_la_rampe(), REPETITION, rang)


def niveau_de_la_raie(rang: int) -> float:
    """Le module de la raie, en decibels."""
    return 20 * math.log10(abs(complex(raie_de_l_onde_periodique(rang))))


def puissance(periode: object) -> sp.Expr:
    """La puissance moyenne de la salve repetee toutes les `periode` secondes."""
    salve = sp.Piecewise((AMPLITUDE * sp.sin(2000 * sp.pi * T), T < SALVE), (0, True))
    return puissance_moyenne_d_un_signal(salve, sp.sympify(periode))


def periode_pour_une_puissance_unite() -> sp.Expr:
    """Le `T` pour lequel la puissance vaut un."""
    inconnue = sp.Symbol("T", positive=True)
    energie = puissance(1)
    (solution,) = sp.solve(sp.Eq(energie / inconnue, 1), inconnue)
    return solution


def valeur_efficace(periode: object) -> sp.Expr:
    """La racine de la puissance."""
    return sp.sqrt(puissance(periode))


def facteur_pour_diviser_la_valeur_efficace(diviseur: int) -> int:
    """La puissance va comme `1/T` : la valeur efficace comme `1/racine(T)`."""
    return diviseur**2
