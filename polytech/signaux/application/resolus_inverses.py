"""Exercices resolus, 0.17 a 0.21 : des inverses de Laplace.

Chaque image est une fraction rationnelle : on la decompose en fractions
simples, et chaque pole donne une exponentielle. Un numerateur de meme
degre que le denominateur laisse d'abord une impulsion de Dirac.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.signaux.domain.heaviside import original
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T
from polytech.signaux.domain.transformees import transformee_de_laplace

IMAGES: Final[dict[str, sp.Expr]] = {
    "0.17a": 1 / (P**2 + 3 * P + 2),
    "0.17b": 1 / (P**2 + 4),
    "0.17c": 1 / (P**2 * (P + 10)),
    "0.18": (P + 1) / (P**2 * (P + 10)),
    "0.19a": P**2 / (P**2 + 6 * P + 8),
    "0.19b": (P + 1) / ((P + 3) ** 2 * (P + 10)),
    "0.20": (P**3 + P**2 + 2 * P + 1) / (P**3 + 3 * P**2 + P + 3),
    "0.21": 1 / ((P**2 + 4) * (P**2 - 4)),
}
"""Les images des enonces ; celle de 0.18 est donnee sans son retard."""
RETARD: Final = 5
"""Le facteur `e^(-5p)` de 0.18 retarde l'original de cinq secondes."""


def original_de(nom: str) -> sp.Expr:
    """L'original de l'image de ce nom, pour `t >= 0`."""
    return original(IMAGES[nom])


def image_de_l_original(nom: str) -> sp.Expr:
    """La transformee de Laplace de l'original : elle doit rendre l'image."""
    return transformee_de_laplace(original_de(nom))


def original_retarde() -> sp.Expr:
    """0.18 : l'original sans retard, ou `t` est remplace par `t - 5`."""
    return original_de("0.18").subs(T, T - RETARD)


def forme_de_la_solution_20() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """0.20 : le coefficient de `e^(-3t)`, puis l'amplitude et la phase du cosinus."""
    sans_impulsion = original_de("0.20").subs(sp.DiracDelta(T), 0)
    developpe = sp.expand(sans_impulsion)
    exponentielle = developpe.coeff(sp.exp(-3 * T))
    en_cosinus, en_sinus = developpe.coeff(sp.cos(T)), developpe.coeff(sp.sin(T))
    amplitude = sp.sqrt(en_cosinus**2 + en_sinus**2)
    return exponentielle, amplitude, sp.atan2(en_sinus, en_cosinus)


def poles_a_droite(nom: str) -> tuple[sp.Expr, ...]:
    """Les poles de partie reelle positive : ceux qui font diverger l'original."""
    poles = sp.roots(sp.fraction(IMAGES[nom])[1], P)
    return tuple(pole for pole in poles if sp.re(pole) > 0)
