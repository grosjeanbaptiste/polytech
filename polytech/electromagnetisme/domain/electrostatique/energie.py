"""L'energie potentielle electrostatique d'un systeme de charges.

Le syllabus d'electricite generale (2.21) :

* deux charges : `W_e = q_1 q_2 / (4 pi eps_0 R_12)` (2.21.3), une energie
  **mutuelle**, nulle a l'infini ;
* un systeme : chaque couple compte une fois (2.21.5), soit
  `W_e = (1/2) somme q_i V_i` (2.21.6) ;
* une file infinie de charges alternees `+q`, `-q` de pas `d` (exemple
  2.21.1) : `W_e = -n q^2 2 ln 2 / (4 pi eps_0 d)`, soit `2 ln 2 = 1,386`
  fois l'energie de `n` couples isoles -- la constante de Madelung ;
* une sphere uniformement chargee (exemple 2.21.2) :
  `W_e = (3/5) Q^2 / (4 pi eps_0 a)`.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.potentiel import COULOMB

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.electrostatique.moments import (
        SystemeDeCharges,
    )

MADELUNG_DE_LA_CHAINE: Final = 2 * sp.log(2)
"""`2 ln 2` : ce que les voisins ajoutent a l'energie d'un couple isole."""


def energie_de_deux_charges(
    premiere: object, seconde: object, distance: object
) -> sp.Expr:
    """`W_e = q_1 q_2 / (4 pi eps_0 R_12)`."""
    return COULOMB * sp.sympify(premiere) * sp.sympify(seconde) / distance


def energie_du_systeme(systeme: SystemeDeCharges) -> sp.Expr:
    """La somme des energies mutuelles, chaque couple compte une fois."""
    total = sp.Integer(0)
    for (q_i, r_i), (q_j, r_j) in combinations(systeme.charges, 2):
        distance = sp.sqrt(sum((a - b) ** 2 for a, b in zip(r_i, r_j, strict=True)))
        if distance == 0:
            msg = "charges confondues : leur energie mutuelle est infinie"
            raise ValueError(msg)
        total += energie_de_deux_charges(q_i, q_j, distance)
    return total


def potentiel_sur_la_chaine(charge: object, pas: object, *, voisins: int) -> sp.Expr:
    """Le potentiel sur un site `+q`, du aux `voisins` premiers sites de chaque cote."""
    q = sp.sympify(charge)
    termes = (
        q * (-1) ** rang / (rang * sp.sympify(pas)) for rang in range(1, voisins + 1)
    )
    return 2 * COULOMB * sp.Add(*termes)


def energie_de_la_chaine(couples: object, charge: object, pas: object) -> sp.Expr:
    """`W_e = -n q^2 2 ln 2 / (4 pi eps_0 d)`, pour `n` couples d'une file infinie."""
    isole = energie_de_deux_charges(charge, -sp.sympify(charge), pas)
    return sp.sympify(couples) * MADELUNG_DE_LA_CHAINE * isole


def energie_de_la_sphere(charge: object, rayon: object) -> sp.Expr:
    """`W_e = (3/5) Q^2 / (4 pi eps_0 a)`, pour une sphere chargee en volume."""
    return sp.Rational(3, 5) * COULOMB * sp.sympify(charge) ** 2 / rayon
