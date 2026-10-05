"""La conservation de la masse sous forme locale, en trois dimensions.

Chapitre IV.1 des planches. Pour un systeme ferme la masse est constante ;
pour un volume de controle, `m_in - m_out = dm_CV/dt`. Sous forme locale :

`d(rho)/dt + div(rho v) = 0`,

ce que la derivee materielle ecrit `D(rho)/Dt + rho div v = 0`. Pour un
ecoulement incompressible, il ne reste que `div v = 0`.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import ESPACE, T
from polytech.maths.domain.analyse_vectorielle import divergence, gradient

Vecteur = tuple[sp.Expr, sp.Expr, sp.Expr]


def residu_de_continuite(masse_volumique: sp.Expr, vitesse: Vecteur) -> sp.Expr:
    """`d(rho)/dt + div(rho v)` : nul si la masse se conserve."""
    rho = sp.sympify(masse_volumique)
    flux = tuple(rho * composante for composante in vitesse)
    return sp.simplify(sp.diff(rho, T) + divergence(flux, ESPACE))


def residu_en_derivee_materielle(masse_volumique: sp.Expr, vitesse: Vecteur) -> sp.Expr:
    """`D(rho)/Dt + rho div v` : la meme loi, en suivant la particule."""
    rho = sp.sympify(masse_volumique)
    pentes = zip(vitesse, gradient(rho, ESPACE), strict=True)
    advection = sp.Add(*(v * pente for v, pente in pentes))
    return sp.simplify(sp.diff(rho, T) + advection + rho * divergence(vitesse, ESPACE))
