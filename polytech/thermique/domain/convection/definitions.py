"""Les definitions de la convection : le coefficient et l'equation de l'energie.

Syllabus de thermique, tome 2, premiere partie, sections 1.1 a 1.3.1.
Le coefficient de convection est une **definition**, `phi = K (t_p -
t_f)`. Le fluide etant au repos a la paroi, il s'y comporte comme un
solide, et `K` se lit sur la pente du champ de temperature :

    K = -lambda d/dn [(t - t_f) / (t_p - t_f)], a la paroi.

Le coefficient moyen d'une surface est la moyenne du coefficient local.

Dans un fluide, l'equation de l'energie est celle de Fourier-Kirchhoff
ou la derivee `d/dtau` devient la derivee historique `D/Dtau`, qui suit
l'element de fluide : `Dt/Dtau = a laplacien(t)`. On y a neglige la
dissipation visqueuse et l'echauffement cinetique `u^2 / (2 c_p)`.

En regime turbulent, l'agitation ajoute a la conduction un transport :
`phi = -lambda dt_moy/dy + rho c_p moyenne(v_ag t_ag)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.fluides.domain.turbulence import covariance
from polytech.maths.domain.analyse_vectorielle import gradient
from polytech.thermique.domain.conduction.fourier import (
    ESPACE,
    residu_de_fourier_kirchhoff,
)

if TYPE_CHECKING:
    from polytech.fluides.domain.turbulence import Decomposition

type Grandeur = sp.Expr | float
type Vecteur = tuple[sp.Expr, sp.Expr, sp.Expr]


def coefficient_local(
    champ: sp.Expr,
    normale: sp.Symbol,
    conductivite: Grandeur,
    *,
    paroi: Grandeur,
    fluide: Grandeur,
) -> sp.Expr:
    """`K`, la paroi etant en `normale = 0` et `t_f` uniforme."""
    pente = sp.diff(champ, normale).subs(normale, 0)
    return sp.simplify(-conductivite * pente / (paroi - fluide))


def coefficient_moyen(
    local: sp.Expr, abscisse: sp.Symbol, longueur: Grandeur
) -> sp.Expr:
    """La moyenne de `K` du bord d'attaque a `longueur`."""
    return sp.simplify(sp.integrate(local, (abscisse, 0, longueur)) / longueur)


def residu_de_l_energie(
    champ: sp.Expr, vitesse: Vecteur, diffusivite: Grandeur
) -> sp.Expr:
    """`Dt/Dtau - a laplacien(t)` : nul pour un champ licite."""
    pentes = gradient(sp.sympify(champ), ESPACE)
    advection = sp.Add(*(u * pente for u, pente in zip(vitesse, pentes, strict=True)))
    return sp.simplify(residu_de_fourier_kirchhoff(champ, diffusivite) + advection)


def echauffement_cinetique(vitesse: float, chaleur_massique: float) -> float:
    """`u^2 / (2 c_p)` : ce dont s'echauffe un fluide amene au repos."""
    if chaleur_massique <= 0:
        msg = f"chaleur massique nulle ou negative : {chaleur_massique} J/(kg K)"
        raise ValueError(msg)
    return vitesse**2 / (2 * chaleur_massique)


def densite_de_flux_turbulente(  # noqa: PLR0913 -- le fluide, deux signaux, la conduction
    masse_volumique: float,
    chaleur_massique: float,
    vitesse: Decomposition,
    temperature: Decomposition,
    *,
    conductivite: float,
    gradient: float,
) -> float:
    """La densite de flux moyenne, normale a la paroi, en W/m^2."""
    agitation = masse_volumique * chaleur_massique * covariance(vitesse, temperature)
    return -conductivite * gradient + agitation
