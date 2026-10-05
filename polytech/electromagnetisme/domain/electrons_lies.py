"""Le mouvement des electrons lies, vu a l'echelle macroscopique.

Le syllabus d'electricite generale (1.7) : l'electron lie decrit, en un
temps `T_A`, une orbite fermee autour du noyau. Sa position moyenne est
celle du noyau (1.7.1), sa vitesse moyenne aussi (1.7.2 et 1.7.3), car le
mouvement relatif se referme. La moyenne macroscopique se prend sur une
duree `T` (1.7.4) :

    < . > = (1 / T) integrale de 0 a T de ( . ) dt.

D'ou la regle du cours : remplacer l'electron lie par une charge placee au
centre du noyau -- sauf quand position et vitesse figurent ensemble, « la
moyenne d'un produit n'est pas egale au produit des moyennes ».
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS


def moyenne_temporelle(grandeur: object, duree: object) -> sp.Expr:
    """`(1 / T) integrale de 0 a T` de la grandeur, fonction du temps."""
    periode = sp.sympify(duree)
    if periode.is_positive is not True:
        msg = f"duree strictement positive attendue, et non {periode}"
        raise ValueError(msg)
    integrale = sp.integrate(sp.sympify(grandeur), (TEMPS, 0, periode))
    return sp.simplify(integrale / periode)
