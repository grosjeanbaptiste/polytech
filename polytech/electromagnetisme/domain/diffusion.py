"""La distribution d'un courant variable, et celle d'un flux variable.

Le syllabus d'electricite generale (15.14 et 15.15). Dans un conducteur
homogene au repos, `E = rho i`, `rot E = -mu dH/dt` et `rot H = i`
donnent `laplacien i = (mu / rho) di/dt` (15.14.2). En courant continu le
second membre est nul ; un courant variable, lui, n'est pas reparti
uniformement : il s'etablit par la surface et gagne en profondeur. Dans
un cylindre, `(1 / r) d/dr (r di/dr) = (mu / rho) di/dt` (15.14.4).

L'induction d'un circuit magnetique de permeabilite constante suit la
meme equation (15.15.2) : a conditions identiques, les deux
distributions sont les memes.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.symboles import ABSCISSE, RAYON, TEMPS

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def _second_membre(
    grandeur: sp.Expr, permeabilite: Grandeur, resistivite: Grandeur
) -> sp.Expr:
    return permeabilite / resistivite * sp.diff(grandeur, TEMPS)


def residu_de_diffusion_plane(
    grandeur: sp.Expr,
    permeabilite: Grandeur,
    resistivite: Grandeur,
) -> sp.Expr:
    """`d2/dx2 - (mu / rho) d/dt` : (15.14.2) quand tout ne depend que de `x`."""
    laplacien = sp.diff(grandeur, ABSCISSE, 2)
    return sp.simplify(laplacien - _second_membre(grandeur, permeabilite, resistivite))


def residu_de_diffusion_cylindrique(
    grandeur: sp.Expr,
    permeabilite: Grandeur,
    resistivite: Grandeur,
) -> sp.Expr:
    """`(1 / r) d/dr (r d/dr) - (mu / rho) d/dt` (15.14.4)."""
    laplacien = sp.diff(RAYON * sp.diff(grandeur, RAYON), RAYON) / RAYON
    return sp.simplify(laplacien - _second_membre(grandeur, permeabilite, resistivite))
