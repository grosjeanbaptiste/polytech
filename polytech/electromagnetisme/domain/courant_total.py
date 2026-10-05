"""Le courant de deplacement et le courant total.

Le syllabus d'electricite generale (14.7, 14.9 et 14.21). En remplacant
la charge par `div D` dans l'equation de continuite, on trouve que
`i_cd + i_cv + dD/dt` est a flux conservatif : c'est lui que le
rotationnel du champ magnetique egale (14.7.1). `i_d = dD/dt` (14.9.1)
est le courant de deplacement, et la somme des trois le courant total
(14.9.2), dont les lignes sont fermees (14.9.3).

Le courant de deplacement existe dans le vide ; dans la matiere il porte
en outre le courant de polarisation, un vrai mouvement de charges
(14.18.1). A la surface d'un conducteur, le tube de courant de conduction
se prolonge dans l'isolant par un tube de courant de deplacement de meme
intensite (14.21.6). Dans le conducteur lui-meme il est negligeable : son
amplitude rapportee a celle du courant de conduction vaut `2 pi f tau`,
`tau` etant le temps de relaxation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from polytech.electromagnetisme.domain.symboles import TEMPS

if TYPE_CHECKING:
    Vecteur = tuple[Any, Any, Any]


def courant_de_deplacement(deplacement: Vecteur) -> Vecteur:
    """`i_d = dD/dt` (14.9.1), en amperes par metre carre."""
    x, y, z = (sp.diff(sp.sympify(composante), TEMPS) for composante in deplacement)
    return (x, y, z)


def courant_total(
    conduction: Vecteur, convection: Vecteur, deplacement: Vecteur
) -> Vecteur:
    """`i = i_cd + i_cv + dD/dt` (14.9.2)."""
    varie = courant_de_deplacement(deplacement)
    x, y, z = (a + b + c for a, b, c in zip(conduction, convection, varie, strict=True))
    return (x, y, z)


def rapport_du_deplacement_a_la_conduction(frequence: Any, relaxation: Any) -> Any:  # noqa: ANN401 -- reels ou symboles
    """`2 pi f tau` : l'amplitude de `i_d` rapportee a celle de `i_cd`."""
    return 2 * sp.pi * frequence * relaxation
