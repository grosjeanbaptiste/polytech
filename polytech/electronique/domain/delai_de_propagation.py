"""Ce qu'un delai de propagation coute a une tension sinusoidale.

Syllabus, section 2.8.5. Au bout d'une ligne de longueur `l`, la tension
`A sin(w t)` arrive retardee : `A sin(w t - 2 pi l / lambda)`. L'ecart
entre la sortie et l'entree est, par l'identite `sin a - sin b = 2 sin((a
- b) / 2) cos((a + b) / 2)`,

    V_out - V_in = 2 A sin(-pi l / lambda) cos(w t - pi l / lambda),

une sinusoide d'amplitude `2 A sin(pi l / lambda)`. A la limite d'usage
`l = lambda / 20`, elle vaut 31 % de `A`.
"""

from __future__ import annotations

import math


def amplitude_de_l_ecart(longueur: float, longueur_d_onde: float) -> float:
    """`2 |sin(pi l / lambda)|`, rapportee a l'amplitude du signal."""
    if longueur_d_onde <= 0:
        msg = f"longueur d'onde nulle ou negative : {longueur_d_onde} m"
        raise ValueError(msg)
    if longueur < 0:
        msg = f"longueur de ligne negative : {longueur} m"
        raise ValueError(msg)
    return 2 * abs(math.sin(math.pi * longueur / longueur_d_onde))


def retard_de_phase_en_degres(longueur: float, longueur_d_onde: float) -> float:
    """`pi l / lambda` en degres : le dephasage de `cos(w t - pi l / lambda)`."""
    amplitude_de_l_ecart(longueur, longueur_d_onde)  # memes gardes d'entree
    return 180.0 * longueur / longueur_d_onde


def pire_ecart_sur_une_periode(
    longueur: float, longueur_d_onde: float, points: int = 360
) -> float:
    """Le plus grand `|sin(t - 2 pi l / lambda) - sin t|` releve sur une periode."""
    if points <= 0:
        msg = f"nombre de points non positif : {points}"
        raise ValueError(msg)
    retard = 2 * math.pi * longueur / longueur_d_onde
    amplitude_de_l_ecart(longueur, longueur_d_onde)  # memes gardes d'entree
    instants = (2 * math.pi * k / points for k in range(points))
    return max(abs(math.sin(t - retard) - math.sin(t)) for t in instants)
