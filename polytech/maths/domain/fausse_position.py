"""La methode de la fausse position, ou « regula falsi ».

Une fonction continue qui change de signe entre deux bornes s'y annule.
La dichotomie coupe l'intervalle en son milieu ; la fausse position le
coupe la ou la **corde** joignant les deux extremites croise l'axe :

    c = (a f(b) - b f(a)) / (f(b) - f(a)),

puis garde le cote ou le signe change. Elle converge toujours, et plus
vite que la dichotomie quand la fonction est proche d'une droite.

Sur une fonction convexe, une des deux bornes ne bouge plus et la
convergence ralentit. La variante dite de l'Illinois, retenue ici,
divise par deux l'ordonnee de la borne restee immobile : la corde
bascule, et la borne finit par se deplacer.

Le syllabus de thermique (tome 2, page 110) la nomme pour resoudre la
continuite du flux en la temperature de paroi ; la methode est d'analyse
numerique, et vit donc ici.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

TOLERANCE = 1e-12
ITERATIONS = 200


def _corde(bas: float, haut: float, en_bas: float, en_haut: float) -> float:
    return (bas * en_haut - haut * en_bas) / (en_haut - en_bas)


def racine_par_fausse_position(
    fonction: Callable[[float], float],
    bas: float,
    haut: float,
    *,
    tolerance: float = TOLERANCE,
    iterations: int = ITERATIONS,
) -> float:
    """La racine de `fonction` entre `bas` et `haut`, ou elle change de signe."""
    en_bas, en_haut = fonction(bas), fonction(haut)
    if en_bas * en_haut > 0:
        msg = f"pas de changement de signe entre {bas} et {haut}"
        raise ValueError(msg)
    for _ in range(iterations):
        milieu = _corde(bas, haut, en_bas, en_haut)
        au_milieu = fonction(milieu)
        if abs(au_milieu) <= tolerance:
            return milieu
        if en_bas * au_milieu < 0:
            haut, en_haut = milieu, au_milieu
            en_bas /= 2  # variante de l'Illinois : la borne immobile est allegee
        else:
            bas, en_bas = milieu, au_milieu
            en_haut /= 2
    msg = f"pas de convergence en {iterations} iterations"
    raise ValueError(msg)
