"""La resonance d'une branche RLC, et l'antiresonance du circuit bouchon.

Le syllabus d'electricite generale (16.21). En serie, l'impedance passe
par son minimum `R` a la pulsation `omega_r = 1 / racine(L C)` (16.21.5).
Rapportee a `R`, elle s'ecrit `z_s = 1 + j Q (x - 1 / x)` (16.21.7), ou
`x = omega / omega_r` et `Q = omega_r L / R` est le facteur de qualite
(16.21.9) -- aussi facteur de surtension (16.21.14). Elle vaut
`racine(2)` aux deux pulsations de coupure (16.21.11) et (16.21.12),
distantes de `1 / Q` (16.21.13) : la puissance dissipee y est moitie.

Un condensateur en parallele avec `R` et `L` en serie donne
`z_p = (1 + j Q x) / (1 - x^2 + j x / Q)` (16.21.19). Son module passe par
un maximum (16.21.21) a la pulsation d'antiresonance (16.21.20), proche
de `omega_r` des que `Q` est grand : c'est le circuit bouchon, dont
l'impedance vaut alors `Q^2 R`.
"""

from __future__ import annotations

import sympy as sp

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""


def impedance_reduite_serie(qualite: Grandeur, reduite: Grandeur) -> sp.Expr:
    """`z_s = 1 + j Q (x - 1 / x)` (16.21.7)."""
    abscisse = sp.sympify(reduite)
    return 1 + sp.I * qualite * (abscisse - 1 / abscisse)


def _inverse(qualite: Grandeur) -> sp.Expr:
    if (sp.sympify(qualite) <= 0) == sp.true:
        msg = f"facteur de qualite nul ou negatif : {qualite}"
        raise ValueError(msg)
    return 1 / sp.sympify(qualite)


def pulsations_de_coupure(qualite: Grandeur) -> tuple[sp.Expr, sp.Expr]:
    """`(x', x'')` de (16.21.12) et (16.21.11) : la ou `z_s` vaut `racine(2)`."""
    inverse = _inverse(qualite)
    racine = sp.sqrt(inverse**2 + 4)
    return (racine - inverse) / 2, (racine + inverse) / 2


def impedance_reduite_parallele(qualite: Grandeur, reduite: Grandeur) -> sp.Expr:
    """`z_p = (1 + j Q x) / (1 - x^2 + j x / Q)` (16.21.19)."""
    denominateur = 1 - reduite**2 + sp.I * reduite * _inverse(qualite)
    return (1 + sp.I * qualite * reduite) / denominateur


def pulsation_d_antiresonance(qualite: Grandeur) -> sp.Expr:
    """`x_a = racine(racine(1 + 2 / Q^2) - 1 / Q^2)` (16.21.20)."""
    inverse = _inverse(qualite)
    carre = sp.sqrt(1 + 2 * inverse**2) - inverse**2
    if (carre <= 0) == sp.true:
        msg = f"pas d'antiresonance pour Q = {qualite} : circuit trop amorti"
        raise ValueError(msg)
    return sp.sqrt(carre)


def maximum_a_l_antiresonance(qualite: Grandeur) -> sp.Expr:
    """`z_p,max = Q / racine(2 racine(1 + 2 / Q^2) - (2 + 1 / Q^2))` (16.21.21)."""
    inverse = _inverse(qualite)
    sous_la_racine = 2 * sp.sqrt(1 + 2 * inverse**2) - (2 + inverse**2)
    return qualite / sp.sqrt(sous_la_racine)
