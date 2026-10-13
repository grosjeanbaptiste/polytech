"""Primitiver une somme : fractions simples, morceaux connus, derivee reconnue.

Trois gestes qui reecrivent l'integrande avant de le primitiver. Chacun
verifie sa reecriture, et `par_morceaux` verifie en plus chaque primitive
qu'on lui donne : la ou `sympy` ne sait pas integrer un terme, c'est la
derivee qui fait foi.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.application.recueil_primitives_gestes import (
    POINTS,
    SEUIL,
    Points,
    ecart,
    valeur_absolue,
)
from polytech.maths.application.resolution_exs_01 import Resolution
from polytech.maths.domain.symboles import VARIABLE as X

if TYPE_CHECKING:
    from collections.abc import Sequence

FRACTIONS = "decomposition en fractions simples"


def fractions_simples(integrande: sp.Expr) -> Resolution:
    """Une fraction rationnelle, decomposee puis primitivee terme a terme."""
    decomposee = sp.apart(integrande, X)
    termes = sp.Add.make_args(decomposee)
    primitive = sp.Add(*[sp.integrate(terme, X) for terme in termes])
    return Resolution(integrande, FRACTIONS, (decomposee,), valeur_absolue(primitive))


def par_morceaux(
    integrande: sp.Expr, morceaux: Sequence[tuple[sp.Expr, sp.Expr]], methode: str
) -> Resolution:
    """L'integrande est la somme des termes ; chacun vient avec sa primitive."""
    for terme, primitive in morceaux:
        if ecart(sp.diff(primitive, X), terme) > SEUIL:
            msg = f"{primitive} n'est pas une primitive du terme {terme}"
            raise ValueError(msg)
    somme = sp.Add(*[terme for terme, _ in morceaux], evaluate=False)
    if ecart(sp.Add(*[terme for terme, _ in morceaux]), integrande) > SEUIL:
        msg = f"les termes {somme} ne redonnent pas l'integrande {integrande}"
        raise ValueError(msg)
    primitive = sp.Add(*[primitive for _, primitive in morceaux])
    return Resolution(integrande, methode, (somme,), primitive)


def reconnue(
    integrande: sp.Expr,
    reecrit: sp.Expr,
    primitive: sp.Expr,
    methode: str,
    points: Points = POINTS,
) -> Resolution:
    """Une reecriture ou la derivee d'une fonction connue se reconnait."""
    if ecart(reecrit, integrande, points) > SEUIL:
        msg = f"la reecriture {reecrit} ne vaut pas l'integrande {integrande}"
        raise ValueError(msg)
    if ecart(sp.diff(primitive, X), integrande, points) > SEUIL:
        msg = f"{primitive} n'est pas une primitive de {integrande}"
        raise ValueError(msg)
    return Resolution(integrande, methode, (reecrit,), primitive)
