"""Les moments du second ordre : par rapport aux plans, aux axes, au point.

Dans un systeme d'axes `Oxyz`, on definit :

* par rapport aux **plans** de coordonnees, `I_yz = integrale x^2 dm`,
  `I_zx = integrale y^2 dm`, `I_xy = integrale z^2 dm` ;
* par rapport aux **axes**, `I_x = integrale (y^2 + z^2) dm` et ses deux
  permutations ;
* par rapport au **point** `O`, `I_O = integrale (x^2 + y^2 + z^2) dm` ;
* les **produits d'inertie** `P_xy = integrale x y dm`, `P_yz`, `P_xz`.

Ils sont lies : `I_x = I_xz + I_xy`, et
`I_O = I_xy + I_yz + I_zx = (I_x + I_y + I_z) / 2`. Le **rayon de
giration** `k` par rapport a un axe est defini par `m k^2 = I_a`.

Source : `MecaniqueRationnelle2/SYLLABUS/Syllabus_Theorie_2025_2026.pdf`,
section 6.2, pages 105 et 106, relations (6.12) a (6.25).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from polytech.maths.domain.algebre.produit_vectoriel import Vecteur
    from polytech.mecanique.domain.inertie.repartition import Repartition


def moments_planaires(solide: Repartition) -> Vecteur:
    """`(I_yz, I_zx, I_xy)`, relations (6.12) a (6.14)."""
    x, y, z = solide.position
    return solide.integrale(x**2), solide.integrale(y**2), solide.integrale(z**2)


def moments_axiaux(solide: Repartition) -> Vecteur:
    """`(I_x, I_y, I_z)`, relations (6.15) a (6.17)."""
    x, y, z = solide.position
    return (
        solide.integrale(y**2 + z**2),
        solide.integrale(x**2 + z**2),
        solide.integrale(x**2 + y**2),
    )


def moment_polaire(solide: Repartition) -> sp.Expr:
    """`I_O`, relation (6.18)."""
    x, y, z = solide.position
    return solide.integrale(x**2 + y**2 + z**2)


def produits_d_inertie(solide: Repartition) -> Vecteur:
    """`(P_xy, P_yz, P_xz)`, relations (6.19) a (6.21)."""
    x, y, z = solide.position
    return solide.integrale(x * y), solide.integrale(y * z), solide.integrale(x * z)


def rayon_de_giration(moment: sp.Expr, masse: sp.Expr) -> sp.Expr:
    """`k`, tel que `m k^2 = I_a` ; leve si la masse est nulle."""
    if masse == 0:
        msg = "masse nulle : le rayon de giration n'est pas defini"
        raise ValueError(msg)
    return sp.sqrt(moment / masse)
