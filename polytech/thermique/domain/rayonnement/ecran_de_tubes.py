"""L'ecran de tubes face a un plan rayonnant.

Syllabus de thermique, tome 1, deuxieme partie, section 2.3.4. Une
rangee de tubes de diametre `d`, au pas `l`, forme un ecran devant un
plan `P`. La regle des elastiques donne l'aire d'echange entre deux
tubes voisins, par metre de longueur : un segment croise moins un
segment non croise,

    12 = sqrt(l^2 - d^2) + d arcsin(d / l) - l.

Le syllabus imprime cette aire avec un prefacteur double. Les cordes
ne donnent que `d`, et c'est avec lui que la suite retrouve la courbe 1
de la figure 2.15 -- et qu'un ecran de tubes jointifs arrete tout.

Un tube voit ses deux voisins et les deux plans qui ferment le
systeme : `2 F_12 + 2 F_1P = 1`. La reciprocite donne alors
`F_PT = (pi d / l) F_1P`. Une seconde rangee, en quinconce, recoit
directement ce que la premiere laisse passer.
"""

from __future__ import annotations

import math


def aire_entre_deux_tubes(diametre: float, pas: float) -> float:
    """`12`, en m^2 par metre de tube."""
    if pas < diametre:
        msg = f"tubes qui se chevauchent : pas {pas} m, diametre {diametre} m"
        raise ValueError(msg)
    tangente = math.sqrt(pas**2 - diametre**2)
    return tangente + diametre * math.asin(diametre / pas) - pas


def facteur_d_un_tube_vers_le_plan(diametre: float, pas: float) -> float:
    """`F_1P = (1 - 2 F_12) / 2`."""
    vers_le_voisin = aire_entre_deux_tubes(diametre, pas) / (math.pi * diametre)
    return (1 - 2 * vers_le_voisin) / 2


def facteur_du_plan_vers_la_rangee(diametre: float, pas: float) -> float:
    """`F_PT`, la part du rayonnement du plan que la rangee intercepte."""
    vers_le_plan = facteur_d_un_tube_vers_le_plan(diametre, pas)
    return math.pi * diametre / pas * vers_le_plan


def facteur_direct_vers_la_seconde_rangee(diametre: float, pas: float) -> float:
    """Ce qui traverse la premiere rangee et s'arrete sur la seconde."""
    premiere = facteur_du_plan_vers_la_rangee(diametre, pas)
    return (1 - premiere) * premiere
