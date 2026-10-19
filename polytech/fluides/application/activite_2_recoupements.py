"""Activite d'integration 2 : chaque resultat refait par un schema numerique.

Le corrige tire ses profils et ses lois du calcul formel (`sympy`). Ici,
`scipy` les refait sans rien savoir des formes closes : une resolution aux
limites pour le film, une quadrature pour son debit, un Runge--Kutta pour
le remplissage. Chaque fonction rend le plus grand ecart, en valeur
absolue, entre les deux chemins.
"""

from __future__ import annotations

import math

import numpy as np
import sympy as sp
from scipy.integrate import quad, solve_bvp, solve_ivp

from polytech.fluides.application import activite_2 as a2
from polytech.fluides.domain.grandeurs import PESANTEUR

_NOEUDS = 101
_TOLERANCE = 1e-12


def _profil_exact(abscisse: float) -> float:
    return float(a2.profil_normalise().subs(a2.ABSCISSE_REDUITE, abscisse))


def ecart_du_profil() -> float:
    """`w*'' = 1`, `w*(0) = 0`, `w*'(1) = 0`, resolue par collocation."""
    maillage = np.linspace(0.0, 1.0, _NOEUDS)
    solution = solve_bvp(
        lambda _x, y: np.vstack((y[1], np.ones_like(y[0]))),
        lambda gauche, droite: np.array([gauche[0], droite[1]]),
        maillage,
        np.zeros((2, maillage.size)),
        tol=_TOLERANCE,
    )
    if not solution.success:
        msg = f"le solveur a echoue : {solution.message}"
        raise RuntimeError(msg)
    return max(abs(float(solution.sol(x)[0]) - _profil_exact(x)) for x in maillage)


def ecart_du_debit() -> float:
    """L'aire sous le profil normalise, par quadrature, contre `-1/3`."""
    aire, _erreur = quad(lambda x: x * (x - 2) / 2, 0.0, 1.0)
    formule = sp.integrate(a2.profil_normalise(), (a2.ABSCISSE_REDUITE, 0, 1))
    return abs(float(aire) - float(formule))


def niveau_par_runge_kutta(fraction: float) -> float:
    """Le niveau `z*` atteint a l'instant ou la forme close annonce `fraction`."""
    if not 0 < fraction < 1:
        msg = f"fraction du niveau maximal hors de ]0, 1[ : {fraction}"
        raise ValueError(msg)
    duree = float(a2.duree_reduite(fraction))
    solution = solve_ivp(
        lambda _t, z: [2 * (1 - math.sqrt(max(z[0], 0.0)))],
        (0.0, duree),
        [0.0],
        method="RK45",
        rtol=_TOLERANCE,
        atol=_TOLERANCE,
    )
    if not solution.success:
        msg = f"le solveur a echoue : {solution.message}"
        raise RuntimeError(msg)
    return float(solution.y[0, -1])


def ecart_du_remplissage() -> float:
    """Le plus grand ecart sur trois niveaux : 50, 90 et 99 % de `z_max`."""
    return max(abs(niveau_par_runge_kutta(f) - f) for f in (0.5, 0.9, 0.99))


def ecart_de_la_puissance() -> float:
    """La puissance a l'arbre par la formule en lettres, contre le domaine."""
    rho, q, d, dz, pertes, eta, g = sp.symbols("rho Q D dz h_L eta g", positive=True)
    hauteur = dz + pertes + 8 * q**2 / (sp.pi**2 * g * d**4)
    formule = rho * g * q * hauteur / eta
    valeur = formule.subs(
        {
            rho: a2.EAU_DE_MER,
            q: a2.DEBIT,
            d: a2.DIAMETRE_DE_LA_BUSE,
            dz: a2.HAUTEUR_DE_LA_BUSE,
            pertes: a2.PERTES,
            eta: a2.RENDEMENT,
            g: PESANTEUR,
        }
    )
    arbre = a2.bateau_pompe().arbre
    return abs(float(valeur) - arbre) / arbre
