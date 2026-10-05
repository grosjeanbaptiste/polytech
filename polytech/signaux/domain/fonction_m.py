"""La fonction `M` et ce qu'elle dit de la reponse indicielle.

Planches « Performances » du cours de signaux et systemes. On pose
`M = L / (1 + L)`, `L` etant la boucle ouverte `G_s C G_1 G_2` : le
capteur etant un gain, `T_R = M / G_s`, et `M` est l'image de la boucle
fermee. Elle ne depend que de `L` ; les abaques de Black-Nichols la
lisent sur la courbe de Black, et quatre nombres la resument :

* `|M(0)|`, inferieur a un sans integrateur, egal a un sinon ; il fixe
  la valeur finale, `y(infini) = |M(0)| R^0 / G_s` ;
* la bande passante, a -3 dB sous `|M(0)|` ; elle encadre le temps de
  montee, `0,3 < t_m BP(Hz) < 0,45` ;
* `|M|max` ; pour un systeme faiblement amorti, le depassement vaut a
  peu pres `9 + 6 (|M|max - |M(0)|)` pour cent, l'ecart etant en dB ;
* `|M|min`, s'il y a un creux, qui annonce un trainage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P

PULSATIONS = np.logspace(-4, 3, 700_001)
MOINS_TROIS_DECIBELS = 10 ** (-3 / 20)
BORNES_DE_MONTEE = (0.3, 0.45)
"""`t_m BP`, la bande passante etant en hertz."""


def fonction_m(ouverte: sp.Expr) -> sp.Expr:
    """`M = L / (1 + L)`."""
    return sp.cancel(ouverte / (1 + ouverte))


def _decibels(module: float) -> float:
    return 20 * math.log10(module)


@dataclass(frozen=True, slots=True)
class CaracteristiquesDeM:
    """`|M(0)|`, `|M|max`, la bande passante en rad/s, et `|M|min` s'il existe."""

    en_continu: float
    maximum: float
    bande_passante: float
    minimum: float | None

    def en_continu_en_decibels(self) -> float:
        """`20 log |M(0)|`."""
        return _decibels(self.en_continu)

    def maximum_en_decibels(self) -> float:
        """`20 log |M|max`."""
        return _decibels(self.maximum)


def caracteristiques_de_m(ouverte: sp.Expr) -> CaracteristiquesDeM:
    """Les quatre nombres, lus sur la reponse frequentielle de `M`."""
    boucle = sp.lambdify(P, ouverte, "numpy")(1j * PULSATIONS)
    modules = np.abs(boucle / (1 + boucle))
    sommet = int(modules.argmax())
    sous = np.nonzero(modules < modules[0] * MOINS_TROIS_DECIBELS)[0]
    creux = int(modules[: sommet + 1].argmin())
    interieur = 0 < creux < sommet and modules[creux] < modules[0]
    return CaracteristiquesDeM(
        en_continu=float(modules[0]),
        maximum=float(modules[sommet]),
        bande_passante=float(PULSATIONS[sous[0]]),
        minimum=float(modules[creux]) if interieur else None,
    )


def valeur_finale_estimee(
    m: CaracteristiquesDeM, *, consigne: float, capteur: float
) -> float:
    """`y(infini) = |M(0)| R^0 / G_s`."""
    return m.en_continu * consigne / capteur


def depassement_estime(ecart_en_decibels: float) -> float:
    """`D(%) = 9 + 6 (|M|max - |M(0)|)`, pour un systeme faiblement amorti."""
    if ecart_en_decibels < 0:
        msg = f"ecart negatif : {ecart_en_decibels} dB -- un maximum depasse |M(0)|"
        raise ValueError(msg)
    return 9 + 6 * ecart_en_decibels


def temps_de_montee_estime(bande_en_hertz: float) -> tuple[float, float]:
    """Les deux bornes de `t_m`, en secondes."""
    if bande_en_hertz <= 0:
        msg = f"bande passante nulle ou negative : {bande_en_hertz} Hz"
        raise ValueError(msg)
    bas, haut = BORNES_DE_MONTEE
    return (bas / bande_en_hertz, haut / bande_en_hertz)
