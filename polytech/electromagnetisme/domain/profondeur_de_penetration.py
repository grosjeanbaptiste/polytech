"""La demonstration de l'effet pelliculaire, pour une surface plane.

Les planches IV.4 (30 et 31) et la note manuscrite qui les accompagne.
Dans un conducteur lineaire, homogene, immobile, ou le courant de
deplacement est negligeable, le courant volumique suit l'equation de
diffusion `d2j/dz2 = (mu / rho) dj/dt`, `z` etant la profondeur sous la
surface. En regime sinusoidal, son substitut complexe verifie
`d2j/dz2 = j omega (mu / rho) j`, dont l'equation caracteristique a pour
racines `+-(1 + j) / delta`, avec la profondeur de penetration
`delta = racine(2 rho / (omega mu))`.

Le courant devant rester fini en profondeur, seule la racine negative
subsiste : `j = j_0 exp(-(1 + j) z / delta)`. La valeur efficace decroit
donc en `exp(-z / delta)`, et le courant retarde d'un radian par
profondeur de penetration.
"""

from __future__ import annotations

import sympy as sp

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def profondeur_de_penetration(
    resistivite: Grandeur, pulsation: Grandeur, permeabilite: Grandeur
) -> sp.Expr:
    """`delta = racine(2 rho / (omega mu))`, en metres."""
    facteurs = tuple(sp.sympify(f) for f in (resistivite, pulsation, permeabilite))
    if any(facteur.is_positive is not True for facteur in facteurs):
        msg = f"resistivite, pulsation ou permeabilite nulle ou negative : {facteurs}"
        raise ValueError(msg)
    return sp.sqrt(2 * facteurs[0] / (facteurs[1] * facteurs[2]))


def _exiger_une_peau(peau: Grandeur) -> sp.Expr:
    valeur = sp.sympify(peau)
    if valeur.is_positive is not True:
        msg = f"profondeur de penetration nulle ou negative : {peau} m"
        raise ValueError(msg)
    return valeur


def racines_caracteristiques(peau: Grandeur) -> tuple[sp.Expr, sp.Expr]:
    """`+-(1 + j) / delta` : les racines de `r^2 = j omega mu / rho`."""
    racine = (1 + sp.I) / _exiger_une_peau(peau)
    return racine, -racine


def retard_de_phase_en_profondeur(profondeur: Grandeur, peau: Grandeur) -> sp.Expr:
    """`phi(z) = z / delta`, en radians."""
    return profondeur / _exiger_une_peau(peau)


def courant_efficace_en_profondeur(
    surface: Grandeur, profondeur: Grandeur, peau: Grandeur
) -> sp.Expr:
    """`j_eff(z) = j_0 exp(-z / delta)`."""
    return surface * sp.exp(-profondeur / _exiger_une_peau(peau))


def substitut_du_courant_en_profondeur(
    surface: Grandeur, profondeur: Grandeur, peau: Grandeur
) -> sp.Expr:
    """`j_0 exp(-(1 + j) z / delta)` : la racine qui reste finie en profondeur."""
    _, decroissante = racines_caracteristiques(peau)
    return surface * sp.exp(decroissante * profondeur)
