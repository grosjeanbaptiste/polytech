"""La sphere uniformement chargee en volume : potentiel et champ partout.

Le syllabus d'electricite generale (exemples 2.6.3 et 2.13.1) : une sphere
de rayon `a` et de charge totale `Q`, soit `eta = Q / (4 pi a^3 / 3)`.

    dehors :  V = Q / (4 pi eps_0 r)                       (2.6.25)
    dedans :  V = (Q / 8 pi eps_0 a) (3 - r^2 / a^2)       (2.6.26)
              E = Q r / (4 pi eps_0 a^3)                   (2.6.28)

Dehors, tout se passe « comme si on avait affaire a une charge ponctuelle
Q localisee au centre » : le champ exterieur vit dans `gauss`. Le
potentiel et le champ sont continus en `r = a`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.potentiel import COULOMB


def potentiel_exterieur(charge: object, distance: object) -> sp.Expr:
    """`V = Q / (4 pi eps_0 r)`, pour `r >= a`."""
    return COULOMB * sp.sympify(charge) / distance


def potentiel_interieur(charge: object, rayon: object, distance: object) -> sp.Expr:
    """`V = (Q / 8 pi eps_0 a) (3 - r^2 / a^2)`, pour `r <= a`."""
    a = sp.sympify(rayon)
    return (
        COULOMB * sp.sympify(charge) / (2 * a) * (3 - sp.sympify(distance) ** 2 / a**2)
    )


def champ_interieur(charge: object, rayon: object, distance: object) -> sp.Expr:
    """`E_r = Q r / (4 pi eps_0 a^3)`, pour `r <= a`."""
    return COULOMB * sp.sympify(charge) * distance / sp.sympify(rayon) ** 3


def deplacement_interieur(charge: object, rayon: object, distance: object) -> sp.Expr:
    """`D_r = Q r / (4 pi a^3)`, pour `r <= a` : il croit lineairement (2.13.10)."""
    return sp.sympify(charge) * distance / (4 * sp.pi * sp.sympify(rayon) ** 3)


def deplacement_radial(densite: object, source: sp.Symbol, distance: object) -> sp.Expr:
    """`D_r(r) = (1 / r^2) integrale de 0 a r de eta(s) s^2 ds` (2.13.9)."""
    enlacee = sp.integrate(sp.sympify(densite) * source**2, (source, 0, distance))
    return sp.simplify(enlacee / sp.sympify(distance) ** 2)
