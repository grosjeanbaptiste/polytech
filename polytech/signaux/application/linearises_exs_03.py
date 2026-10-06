"""Feuille « Systemes », exercices 1 et 2 : lineariser autour d'un point de repos.

Dans les deux cas le geste est le meme : annuler les derivees pour
trouver le repos, puis remplacer chaque terme non lineaire par sa
tangente en ce point. Ce qui reste est lineaire **en variations**.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from polytech.signaux.domain.modeles import coefficients, transmittance_d_une_equation
from polytech.signaux.domain.poles_numeriques import reponse_indicielle
from polytech.signaux.domain.symboles import IMAGE
from polytech.signaux.domain.systemes import (
    linearisation_de_taylor,
    point_de_fonctionnement,
)
from polytech.signaux.domain.transmittance import Transmittance

if TYPE_CHECKING:
    from numpy.typing import NDArray

X, U = sp.symbols("x u", real=True)
COMMANDE_DU_PREMIER: Final = sp.Rational(1, 2)
COMMANDE_DU_SECOND: Final = sp.Integer(1)


def repos_du_premier() -> tuple[sp.Expr, ...]:
    """Derivees nulles dans `x'' + 3x' + 2x^2 = 3u' + u` : `2 x^2 = u`."""
    return point_de_fonctionnement(2 * X**2 - COMMANDE_DU_PREMIER, X)


def transmittance_du_premier(repos: sp.Expr | None = None) -> Transmittance:
    """`dX/dU` autour du repos : `2x^2` y est remplace par sa tangente."""
    point = repos_du_premier()[-1] if repos is None else repos
    pente = linearisation_de_taylor(2 * X**2, X, point).coeff(X, 1)
    return transmittance_d_une_equation(sortie=(1, 3, pente), entree=(3, 1))


def reponses_a_un_petit_echelon(
    instants: NDArray[np.float64], *, amplitude: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """La variation de `x` par le modele global, puis par le modele local.

    L'echelon rend `u'` impulsionnel : on integre `z = x' - 3u`, qui reste
    continu, plutot que `x'`.
    """
    commande = float(COMMANDE_DU_PREMIER) + amplitude
    repos = float(repos_du_premier()[-1])

    def derivees(_: float, etat: NDArray[np.float64]) -> list[float]:
        position, auxiliaire = etat
        return [
            auxiliaire + 3 * commande,
            -3 * (auxiliaire + 3 * commande) - 2 * position**2 + commande,
        ]

    depart = [repos, -3 * float(COMMANDE_DU_PREMIER)]
    bornes = (float(instants[0]), float(instants[-1]))
    solution = solve_ivp(
        derivees, bornes, depart, t_eval=instants, rtol=1e-11, atol=1e-13
    )
    numerateur, denominateur = coefficients(transmittance_du_premier())
    locale = amplitude * reponse_indicielle(numerateur, denominateur, instants)
    return np.asarray(solution.y[0] - repos, dtype=float), locale


def _seconds_membres() -> sp.Matrix:
    """Les trois derivees de l'exercice 2, etat `(x1, x2, y)`."""
    x1, x2, y = sp.symbols("x_1 x_2 y", real=True)
    return sp.Matrix([3 * U**2 - 2 * x1, x1 - x2, 4 * x1 + x2 - 2 * y])


def repos_du_second() -> tuple[sp.Expr, ...]:
    """`(x1, x2, y)` quand les trois derivees s'annulent, pour `u = 1`."""
    etat = sp.symbols("x_1 x_2 y", real=True)
    equations = _seconds_membres().subs(U, COMMANDE_DU_SECOND)
    (solution,) = sp.linsolve(list(equations), etat)
    return tuple(solution)


def transmittance_du_second() -> Transmittance:
    """Equation par equation : `3u^2` linearise, puis trois premiers ordres."""
    pente = linearisation_de_taylor(3 * U**2, U, COMMANDE_DU_SECOND).coeff(U, 1)
    premier = pente / (IMAGE + 2)
    second = premier / (IMAGE + 1)
    return Transmittance(expression=sp.cancel((4 * premier + second) / (IMAGE + 2)))


def transmittance_par_l_etat() -> sp.Expr:
    """`C (pI - A)^-1 B`, avec `A` et `B` jacobiennes au point de repos."""
    etat = sp.Matrix(sp.symbols("x_1 x_2 y", real=True))
    champ = _seconds_membres()
    dynamique = champ.jacobian(etat)
    entree = champ.diff(U).subs(U, COMMANDE_DU_SECOND)
    resolvante = (IMAGE * sp.eye(3) - dynamique).inv()
    return sp.cancel((sp.Matrix([[0, 0, 1]]) * resolvante * entree)[0, 0])


def pente_de_la_caracteristique() -> sp.Expr:
    """La derivee de `y^0(u)`, la sortie de repos en fonction de la commande."""
    etat = sp.symbols("x_1 x_2 y", real=True)
    (solution,) = sp.linsolve(list(_seconds_membres()), etat)
    return sp.diff(solution[2], U).subs(U, COMMANDE_DU_SECOND)
