"""Les facteurs d'angle des geometries rectangulaires, en forme close.

Syllabus de thermique, tome 1, deuxieme partie, section 2.2.3 et
figures 2.5, 2.6, 2.9 et 2.10. Chacun est l'integrale de
`cos(theta_1) cos(theta_2) / (pi r^2)`, que la geometrie permet de
mener a son terme ; les parametres sont des rapports sans dimension.

* Un element `dS_1` et un rectangle `b x c` dans un plan parallele, a la
  distance `a`, un coin a la verticale de l'element : `B = b / a`,
  `C = c / a`.
* Le meme element et un rectangle dans un plan perpendiculaire : `b` est
  le cote parallele au plan de l'element, `c` celui qui s'en eleve.
* Deux rectangles identiques face a face, `X x Y` a la distance `D` :
  `B = X / D`, `C = Y / D`.
* Deux rectangles perpendiculaires d'arete commune `X` : `B = Y / X`
  pour celui qui emet, `C = Z / X` pour celui qui recoit.

Une geometrie qui ne tombe pas sur ces cas s'y ramene par addition et
soustraction d'aires d'echange.
"""

from __future__ import annotations

import math


def _exiger(*rapports: float) -> None:
    for rapport in rapports:
        if rapport <= 0:
            msg = f"rapport nul ou negatif : {rapport}"
            raise ValueError(msg)


def element_vers_rectangle_parallele(b: float, c: float) -> float:
    """Fig. 2.5 : l'element sous un coin du rectangle."""
    _exiger(b, c)
    racine_b, racine_c = math.sqrt(1 + b**2), math.sqrt(1 + c**2)
    premier = b / racine_b * math.atan(c / racine_b)
    second = c / racine_c * math.atan(b / racine_c)
    return (premier + second) / (2 * math.pi)


def element_vers_rectangle_perpendiculaire(b: float, c: float) -> float:
    """Fig. 2.6 : le rectangle se dresse a la distance `a` de l'element."""
    _exiger(b, c)
    racine = math.sqrt(1 + c**2)
    return (math.atan(b) - math.atan(b / racine) / racine) / (2 * math.pi)


def rectangles_paralleles(b: float, c: float) -> float:
    """Fig. 2.9 : deux rectangles identiques, face a face."""
    _exiger(b, c)
    racine_b, racine_c = math.sqrt(1 + b**2), math.sqrt(1 + c**2)
    logarithme = math.log((1 + b**2) * (1 + c**2) / (1 + b**2 + c**2)) / (b * c)
    arcs = -2 / b * math.atan(c) - 2 / c * math.atan(b)
    arcs += 2 / c * racine_c * math.atan(b / racine_c)
    arcs += 2 / b * racine_b * math.atan(c / racine_b)
    return (logarithme + arcs) / math.pi


def rectangles_perpendiculaires(b: float, c: float) -> float:
    """Fig. 2.10 : deux rectangles a angle droit, d'arete commune."""
    _exiger(b, c)
    somme = b**2 + c**2
    arcs = b * math.atan(1 / b) + c * math.atan(1 / c)
    arcs -= math.sqrt(somme) * math.atan(1 / math.sqrt(somme))
    logarithmes = b**2 * math.log((1 + somme) * b**2 / (somme * (1 + b**2)))
    logarithmes += c**2 * math.log((1 + somme) * c**2 / (somme * (1 + c**2)))
    logarithmes -= math.log((1 + somme) / ((1 + b**2) * (1 + c**2)))
    return (arcs + logarithmes / 4) / (math.pi * b)
