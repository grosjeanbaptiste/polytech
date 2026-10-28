"""La stabilite du schema explicite, exactement et asymptotiquement.

Separe de `differences_finies`, qui porte les schemas : ce module-ci
porte leur **stabilite**, et le fichier unique depassait la limite de
taille du depot.

Le syllabus annonce la borne par « on peut montrer qu'il faut » suivi
d'une formule que la conversion n'a pas rendue -- c'etait une image.
Plutot que de l'affirmer sur cette foi, ce module expose les deux bornes
qui existent vraiment, et les tests les verifient par la divergence du
schema :

* `fourier_critique` -- `1 / (2 d)` : la borne **asymptotique**, celle
  qu'un maillage fin approche et celle qu'un dimensionnement emploie,
  parce qu'elle est suffisante a toute finesse ;
* `fourier_critique_du_maillage` -- la borne **exacte** d'une barre de
  `n` noeuds, qui est un peu plus haute : `1,106` fois la premiere sur
  six noeuds, `1,025` sur onze, `1,0001` sur deux cents.

L'ecart vient de ce que la borne asymptotique suppose un mode de
longueur d'onde deux mailles, que le maillage fini ne porte pas tout a
fait : son mode le plus oscillant a un nombre d'onde legerement
inferieur, donc une valeur propre legerement moins negative.
"""

from __future__ import annotations

import math
from typing import Final

DIMENSIONS_DU_COURS: Final = (1, 2, 3)
"""Le cours traite le fil, la plaque et le volume."""

NOEUDS_MINIMAUX: Final = 3
"""Deux bords et un noeud interieur : en dessous il n'y a rien a avancer."""


def fourier_critique(dimensions: int) -> float:
    """`1 / (2 d)` : un demi sur un fil, un quart sur une plaque.

    Borne asymptotique, et borne **suffisante** a toute finesse de
    maillage : c'est celle qu'un dimensionnement emploie.
    """
    if dimensions not in DIMENSIONS_DU_COURS:
        msg = f"dimensions hors de 1 a 3 : {dimensions}"
        raise ValueError(msg)
    return 1.0 / (2.0 * dimensions)


def fourier_critique_du_maillage(noeuds: int) -> float:
    """La borne exacte d'une barre de `noeuds` noeuds, bords tenus.

    Les modes propres du schema valent `1 - 4 Fo sin^2(j pi / 2(n-1))`
    pour `j` de 1 a `n-2`. La stabilite demande que le plus grand reste
    au-dessus de `-1`, d'ou la borne par le mode le plus oscillant.

    Elle tend vers `1/2` quand le maillage se raffine, ce qui est la
    verification de la borne asymptotique -- et l'explication du fait
    qu'un maillage grossier tolere un pas de temps un peu plus grand que
    la regle ne le promet.
    """
    if noeuds < NOEUDS_MINIMAUX:
        msg = f"moins de trois noeuds : {noeuds} -- rien a avancer"
        raise ValueError(msg)
    plus_oscillant = math.sin((noeuds - 2) * math.pi / (2 * (noeuds - 1))) ** 2
    return 1.0 / (2.0 * plus_oscillant)


def est_stable_en_explicite(fourier: float, dimensions: int) -> bool:
    """Vrai si ce nombre de Fourier de maille tient sous la borne du cours."""
    return fourier <= fourier_critique(dimensions)


def avancer_en_explicite(valeurs: list[float], fourier: float) -> list[float]:
    """Un pas d'Euler explicite sur un fil dont les deux bouts sont tenus."""
    interieur = [
        centre + fourier * (gauche - 2 * centre + droite)
        for gauche, centre, droite in zip(
            valeurs, valeurs[1:], valeurs[2:], strict=False
        )
    ]
    return [valeurs[0], *interieur, valeurs[-1]]


def amplitude_maximale_apres(valeurs: list[float], fourier: float, pas: int) -> float:
    """`max |t|` apres `pas` pas : il s'amortit sous la borne, diverge au-dela."""
    for _ in range(pas):
        valeurs = avancer_en_explicite(valeurs, fourier)
    return max(abs(v) for v in valeurs)
