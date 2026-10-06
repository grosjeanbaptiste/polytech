"""Feuille « Boucle fermee - stabilite », exercice 3 : trois boucles ouvertes.

Pour chacune : la stabilite en boucle ouverte se lit sur ses poles ; en
boucle fermee, sur l'equation caracteristique -- sauf pour la deuxieme,
dont le retard n'a pas de polynome et demande un raisonnement en
frequence.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.signaux.domain.simulation_a_delai import indicielle_en_boucle_fermee
from polytech.signaux.domain.stabilite import (
    equation_caracteristique as caracteristique,
)
from polytech.signaux.domain.stabilite_des_poles import nature_de_la_stabilite
from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.transmittance import Transmittance

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

RETARD: Final = 10.0
"""Le retard de la boucle (b), en secondes."""
RATIONNELLES: Final[dict[str, sp.Expr]] = {
    "a": 10 / (P**2 * (1 + 10 * P)),
    "b": 1 / ((1 + P) * (1 + 3 * P)),
    "c": 10 / (P * (1 + P)),
}
"""Les trois boucles ouvertes ; (b) est a multiplier par `exp(-10 p)`."""


def equation_caracteristique(nom: str) -> sp.Expr:
    """`1 + G(p) = 0` en polynome, pour une boucle sans retard."""
    return caracteristique(Transmittance(expression=RATIONNELLES[nom]))


def poles_en_boucle_fermee(nom: str) -> tuple[sp.Expr, ...]:
    """Les racines de l'equation caracteristique."""
    return tuple(sp.Poly(equation_caracteristique(nom), P).all_roots())


def module_de_b(pulsation: float) -> float:
    """`|G(j w)|` pour la boucle (b) : le retard est de module un."""
    return 1 / math.sqrt((1 + pulsation**2) * (1 + 9 * pulsation**2))


def indicielle_de_b(instants: NDArray[np.float64]) -> NDArray[np.float64]:
    """La reponse indicielle de la boucle (b) fermee, retard compris."""
    return indicielle_en_boucle_fermee(RATIONNELLES["b"], RETARD, instants)


def _ouverte(nom: str) -> str:
    """Le cours dit instable une boucle ouverte qui n'est pas strictement stable."""
    nature = nature_de_la_stabilite(RATIONNELLES[nom])
    return "stable" if nature == "strictement stable" else "instable"


def _fermee(nom: str) -> str:
    if nom == "b":
        jamais_au_dessus = all(module_de_b(10**e) < 1 for e in range(-6, 4))
        return "stable" if jamais_au_dessus else "instable"
    a_gauche = all(sp.re(pole) < 0 for pole in poles_en_boucle_fermee(nom))
    return "stable" if a_gauche else "instable"


def verdicts() -> dict[str, tuple[str, str]]:
    """Pour chaque boucle : le verdict en boucle ouverte, puis en boucle fermee."""
    return {nom: (_ouverte(nom), _fermee(nom)) for nom in RATIONNELLES}
