"""L'electrostatique avancee : Poisson, Laplace, images, capacite.

Les planches de ce chapitre tournent autour d'un enonce qu'elles
appellent le « probleme fondamental : unicite de la solution ». C'est lui
qui justifie la methode des **images electriques** : si une configuration
fictive produit les memes conditions aux limites, c'est **la** solution,
et non une solution. La bibliotheque porte donc l'unicite comme une
propriete verifiable, et non comme une remarque.

La garde sur la permittivite relative inferieure a un n'est pas formelle :
un dielectrique renforce toujours la capacite, jamais l'inverse, et une
valeur sous l'unite signale une donnee mal lue.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMITTIVITE_DU_VIDE
from polytech.electromagnetisme.domain.symboles import CARTESIENNES


def laplacien(potentiel: sp.Expr) -> sp.Expr:
    """`d^2V/dx^2 + d^2V/dy^2 + d^2V/dz^2`."""
    return sp.simplify(
        sum(sp.diff(potentiel, variable, 2) for variable in CARTESIENNES),
    )


def equation_de_poisson(potentiel: sp.Expr, densite: sp.Expr) -> sp.Eq:
    """`laplacien(V) = -eta / epsilon_0`."""
    return sp.Eq(laplacien(potentiel), -densite / PERMITTIVITE_DU_VIDE)


def satisfait_laplace(potentiel: sp.Expr) -> bool:
    """Vrai si le laplacien est nul : le potentiel est dit harmonique."""
    return bool(laplacien(potentiel) == 0)


def permittivite(relative: float) -> float:
    """`epsilon = epsilon_r epsilon_0` ; leve sous l'unite."""
    if relative < 1:
        msg = (
            f"permittivite relative sous l'unite : {relative} -- un "
            f"dielectrique renforce la capacite, il ne l'affaiblit pas"
        )
        raise ValueError(msg)
    return relative * PERMITTIVITE_DU_VIDE


def capacite_plane(aire: float, distance: float, relative: float = 1.0) -> float:
    """`C = epsilon S / e` pour deux armatures planes paralleles."""
    if aire <= 0 or distance <= 0:
        msg = f"geometrie invalide : aire {aire} m^2, distance {distance} m"
        raise ValueError(msg)
    return permittivite(relative) * aire / distance


def charge_image(charge: float) -> float:
    """La charge fictive symetrique d'un plan conducteur relie a la terre.

    Elle vaut l'oppose de la charge reelle, placee symetriquement : c'est
    ce qui annule le potentiel sur le plan, donc -- par unicite -- c'est
    la solution du probleme reel dans le demi-espace.
    """
    return -charge


def potentiel_sur_le_plan(charge: float, distance: float) -> float:
    """Le potentiel que la charge et son image produisent sur le plan.

    Il doit valoir exactement zero : c'est la condition aux limites que la
    methode des images sert a satisfaire, et la verifier est la seule
    facon de ne pas croire la methode sur parole.
    """
    if distance <= 0:
        msg = f"distance nulle ou negative : {distance} m"
        raise ValueError(msg)
    facteur = 1 / (4 * float(sp.pi) * PERMITTIVITE_DU_VIDE * distance)
    return facteur * (charge + charge_image(charge))
