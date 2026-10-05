"""Le plan infini uniformement charge.

Le syllabus d'electricite generale le traite deux fois, par l'integrale
directe (exemple 2.10.1) puis par le theoreme de Gauss (exemple 2.13.2),
et trouve les memes resultats :

    E = sgn(z) sigma / (2 eps_0) u_z          (2.10.5), (2.13.13)
    V = -sigma |z| / (2 eps_0)                (2.10.8), (2.13.14)

Le champ ne depend pas de la distance au plan, et saute de `sigma / eps_0`
a sa traversee. Le potentiel, choisi nul sur le plan, ne s'annule pas a
l'infini : « il y a des charges a l'infini ».
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.symboles import COTE


def champ_du_plan(surfacique: object) -> sp.Expr:
    """`E_z = sgn(z) sigma / (2 eps_0)`, le plan charge etant `z = 0`."""
    return sp.sign(COTE) * sp.sympify(surfacique) / (2 * PERMITTIVITE)


def potentiel_du_plan(surfacique: object) -> sp.Expr:
    """`V = -sigma |z| / (2 eps_0)`, nul sur le plan."""
    return -sp.sympify(surfacique) * sp.Abs(COTE) / (2 * PERMITTIVITE)
