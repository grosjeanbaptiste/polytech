"""La turbulence : decomposition de Reynolds, et la loi logarithmique.

Les planches posent la decomposition `u = u_moyen + u'` puis enoncent
leurs « regles sur les moyennes ». Ce sont elles qui font tout le
chapitre, et elles sont ici parce qu'elles se **verifient** : la moyenne
d'une moyenne est cette moyenne, la moyenne d'une fluctuation est nulle,
et la moyenne d'un produit n'est **pas** le produit des moyennes -- d'ou
le tenseur de Reynolds, qui est exactement le terme qui survit.

La loi logarithmique de la sous-couche, `u+ = (1/kappa) ln(y+) + C`, est
donnee avec les constantes de la planche 34 : `kappa = 0,4` et `B = 5,0`.
Ce module a porte `0,41` en les disant illisibles sur les planches ;
elles y sont ecrites, et c'est leur valeur qui est retenue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

KARMAN: Final = 0.4
"""Constante de von Karman : « = 0.4 », planche 34, soit `u+ = 2,5 ln y+ + 5,0`."""

CONSTANTE_LOG: Final = 5.0
"""Constante additive `B` de la loi logarithmique, meme planche."""

LIMITE_SOUS_COUCHE_VISQUEUSE: Final = 5.0
"""`y+ < 5` : la sous-couche visqueuse, ou `u+ = y+`."""

DEBUT_SOUS_COUCHE_LOG: Final = 30.0
"""`y+ > 30` : la zone ou la loi logarithmique vaut."""


@dataclass(frozen=True, slots=True)
class Decomposition:
    """Un signal turbulent, donne par ses echantillons."""

    echantillons: tuple[float, ...]

    def __post_init__(self) -> None:
        """Refuse un echantillonnage vide : sa moyenne n'existe pas."""
        if not self.echantillons:
            msg = "aucun echantillon : la moyenne de Reynolds n'est pas definie"
            raise ValueError(msg)

    def moyenne(self) -> float:
        """La partie moyenne de la decomposition de Reynolds."""
        return sum(self.echantillons) / len(self.echantillons)

    def fluctuations(self) -> tuple[float, ...]:
        """`u' = u - u_moyen`, dont la moyenne doit etre nulle."""
        moyenne = self.moyenne()
        return tuple(valeur - moyenne for valeur in self.echantillons)

    def variance(self) -> float:
        """La moyenne du carre des fluctuations."""
        return sum(f**2 for f in self.fluctuations()) / len(self.echantillons)

    def ecart_type(self) -> float:
        """La racine de la variance, l'intensite de la turbulence."""
        return math.sqrt(self.variance())


def covariance(premier: Decomposition, second: Decomposition) -> float:
    """La moyenne du produit des fluctuations : un terme du tenseur de Reynolds.

    C'est exactement ce qui survit quand on moyenne un produit : la
    moyenne d'un produit n'est pas le produit des moyennes, et l'ecart est
    cette covariance.
    """
    if len(premier.echantillons) != len(second.echantillons):
        msg = (
            f"echantillonnages de tailles differentes : "
            f"{len(premier.echantillons)} et {len(second.echantillons)}"
        )
        raise ValueError(msg)
    paires = zip(premier.fluctuations(), second.fluctuations(), strict=True)
    return sum(un * deux for un, deux in paires) / len(premier.echantillons)


def vitesse_adimensionnee(distance_adimensionnee: float) -> float:
    """`u+` en fonction de `y+` ; leve dans la zone tampon non correlee.

    Entre `y+ = 5` et `y+ = 30` les planches parlent de « sous-couche de
    transition » sans en donner de loi : y appliquer l'une ou l'autre
    serait choisir sans le dire.
    """
    if distance_adimensionnee <= 0:
        msg = f"distance adimensionnee nulle ou negative : {distance_adimensionnee}"
        raise ValueError(msg)
    if distance_adimensionnee < LIMITE_SOUS_COUCHE_VISQUEUSE:
        return distance_adimensionnee
    if distance_adimensionnee < DEBUT_SOUS_COUCHE_LOG:
        msg = (
            f"zone tampon sans loi : {LIMITE_SOUS_COUCHE_VISQUEUSE} <= y+ = "
            f"{distance_adimensionnee} < {DEBUT_SOUS_COUCHE_LOG}"
        )
        raise ValueError(msg)
    return math.log(distance_adimensionnee) / KARMAN + CONSTANTE_LOG
