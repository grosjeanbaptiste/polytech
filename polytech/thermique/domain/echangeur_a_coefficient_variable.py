"""L'echangeur dont le coefficient global varie d'une extremite a l'autre.

Syllabus de thermique, tome 2, deuxieme partie, section 2.4.3. Le
coefficient global n'est presque jamais uniforme : les proprietes des
fluides changent avec la temperature. Tant que la variation reste
faible, la moyenne des deux valeurs d'extremite suffit. Jusqu'a 10 ou
20 %, on la suppose lineaire en l'ecart de temperature,
`K_gl = a + b Dt`, et l'equation s'integre encore :

    Phi = S (K_A Dt_B - K_B Dt_A) / ln(K_A Dt_B / (K_B Dt_A)).     (2.17)

Seules comptent les valeurs de `K_gl` aux deux extremites. Au-dela, on
scinde l'echangeur ou l'on integre numeriquement (section 2.6).
"""

from __future__ import annotations

import math


def coefficient_moyen(premier: float, second: float) -> float:
    """`K_gl,moy = (K_gl,A + K_gl,B) / 2`, pour une variation faible."""
    return (premier + second) / 2


def flux_a_coefficient_lineaire(
    surface: float, extremite_a: tuple[float, float], extremite_b: tuple[float, float]
) -> float:
    """(2.17) ; chaque extremite est `(K_gl, Dt)`, et la limite est `S K Dt`."""
    (premier, ecart_a), (second, ecart_b) = extremite_a, extremite_b
    croise_a, croise_b = premier * ecart_b, second * ecart_a
    if math.isclose(croise_a, croise_b):
        return surface * croise_a
    return surface * (croise_a - croise_b) / math.log(croise_a / croise_b)
