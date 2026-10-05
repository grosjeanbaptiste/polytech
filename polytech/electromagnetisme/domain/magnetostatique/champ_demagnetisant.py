"""Le champ demagnetisant d'un corps aimante.

Le syllabus d'electricite generale (10.16). Un corps plonge dans un champ
applique `H_0` s'aimante, et ses masses magnetiques fictives y creent un
champ `H'` qui s'oppose a `H_0` : le champ d'auto-desaimantation,
homologue du champ depolarisant d'un dielectrique. Le rapport de `H'` a
l'aimantation `M` est le facteur de desaimantation `N`, qui ne depend que
de la forme du corps.

Dans un milieu lineaire, `M = chi H` et `H = H_0 - N M` se referment en
`H = H_0 / (1 + N chi)`. D'ou la remarque finale du paragraphe : le champ
demagnetisant approche le champ applique quand la permeabilite est grande,
et il est negligeable dans un corps diamagnetique ou paramagnetique.
"""

from __future__ import annotations

from polytech.electromagnetisme.domain.magnetostatique.milieux_aimantes import (
    susceptibilite,
)


def champ_demagnetisant(facteur: float, aimantation: float) -> float:
    """`H' = -N M`, en amperes par metre : oppose a l'aimantation."""
    if facteur < 0:
        msg = f"facteur de desaimantation negatif : {facteur}"
        raise ValueError(msg)
    return -facteur * aimantation


def champ_interieur(applique: float, facteur: float, relative: float) -> float:
    """`H = H_0 / (1 + N chi)` : le champ resultant dans un corps lineaire."""
    return applique / (1 - champ_demagnetisant(facteur, susceptibilite(relative)))


def part_du_champ_applique(facteur: float, relative: float) -> float:
    """`|H'| / H_0 = N chi / (1 + N chi)` : tend vers un quand `mu_r` croit."""
    return 1 - champ_interieur(1.0, facteur, relative)
