"""Controle d'homogeneite des resultats de la cinematique (cours III).

L'enonce donne des champs a coefficients numeriques nus : ``u = xt`` n'est
une vitesse que si le 1 sous-entendu porte des secondes a la puissance -2.
Le corrige enonce cette convention ; ce module en verifie la consequence sur
chaque resultat publie, au lieu de la tenir pour acquise.

Un piege s'y cache, et c'est la raison d'etre de
:func:`dimension_des_contributions`. Pour le champ B, ``u`` et ``v`` ne
portent pas le meme coefficient dimensionnel. Sa divergence, ecrite
``t - 1``, semble alors inhomogene -- un temps plus un nombre. Elle ne
l'est pas : le ``t`` vient de ``du/dx`` et porte le coefficient de ``u``,
le ``-1`` vient de ``dv/dy`` et porte celui de ``v``. Il faut donc dimensionner
chaque contribution avec *son* coefficient, jamais la somme d'un bloc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import T, X, Y

if TYPE_CHECKING:
    from collections.abc import Sequence

LONGUEUR = sp.Symbol("L", positive=True)
TEMPS = sp.Symbol("T", positive=True)

VITESSE = LONGUEUR / TEMPS
ACCELERATION = LONGUEUR / TEMPS**2
TAUX = 1 / TEMPS
"""Dimension commune a la divergence, au taux de rotation et au tenseur."""


class ExpressionNonHomogene(Exception):
    """Les termes d'une somme n'ont pas tous la meme dimension."""


class DimensionIndeterminee(Exception):
    """Une expression nulle ne porte aucune dimension : rien a verifier."""


def _dimension_d_un_monome(terme: sp.Expr) -> sp.Expr:
    """Dimension d'un monome, lue sur ses degres -- sans rien substituer.

    Substituer ``x`` et ``y`` par une meme longueur ferait disparaitre
    ``x - y`` par simplification : on compte donc les degres.
    """
    degre_espace = sum(int(sp.degree(terme, variable)) for variable in (X, Y))
    degre_temps = int(sp.degree(terme, T))
    return LONGUEUR**degre_espace * TEMPS**degre_temps


def dimension_de(expression: sp.Expr, coefficient: sp.Expr) -> sp.Expr:
    """Dimension de ``expression`` affectee du ``coefficient`` de sa composante.

    Leve si les termes ne s'accordent pas : c'est le seul resultat utile
    d'un controle d'homogeneite qui echoue.
    """
    termes = [t for t in sp.Add.make_args(sp.expand(expression)) if t != 0]
    if not termes:
        msg = "expression nulle : sa dimension est indeterminee"
        raise DimensionIndeterminee(msg)
    dimensions = {_dimension_d_un_monome(terme) for terme in termes}
    if len(dimensions) != 1:
        msg = (
            "expression non homogene : "
            f"{sorted(map(str, dimensions))} cohabitent dans {expression}"
        )
        raise ExpressionNonHomogene(msg)
    return sp.simplify(coefficient * dimensions.pop())


def dimension_des_contributions(
    contributions: Sequence[tuple[sp.Expr, sp.Expr]],
) -> sp.Expr:
    """Dimension commune a des contributions portant chacune son coefficient.

    Chaque entree est un couple ``(expression, coefficient)``. Le resultat
    n'existe que si toutes retombent sur la meme dimension -- ce qui est
    precisement ce qu'il faut demontrer pour ``div V = t - 1``.
    """
    if not contributions:
        msg = "aucune contribution : dimension indeterminee"
        raise DimensionIndeterminee(msg)
    dimensions = {
        dimension_de(expression, coefficient)
        for expression, coefficient in contributions
        if expression != 0
    }
    if not dimensions:
        msg = "toutes les contributions sont nulles : dimension indeterminee"
        raise DimensionIndeterminee(msg)
    if len(dimensions) != 1:
        msg = (
            f"contributions non homogenes entre elles : {sorted(map(str, dimensions))}"
        )
        raise ExpressionNonHomogene(msg)
    return dimensions.pop()
