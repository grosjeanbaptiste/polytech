"""Mathematique, exercices 1, section 1.2 : la forme d'une integrale definie.

Chaque valeur vient d'une primitive et de ses deux evaluations (Newton-
Leibniz), apres le geste qui rend la primitive accessible : parite,
changement de variable, parties. Les bornes ou la primitive n'est pas
definie -- `t ln t` en 0, par exemple -- sont prises a la limite.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE as X

DEMI_TOUR = sp.pi / 2


@dataclass(frozen=True, slots=True)
class Definie:
    """Un integrande, ses bornes, la methode, la primitive employee, la valeur."""

    integrande: sp.Expr
    bornes: tuple[sp.Expr, sp.Expr]
    methode: str
    primitive: sp.Expr | None
    valeur: sp.Expr


def newton_leibniz(
    integrande: sp.Expr,
    bornes: tuple[sp.Expr, sp.Expr],
    primitive: sp.Expr,
    methode: str,
) -> Definie:
    """`F(b) - F(a)`, chaque borne prise a la limite depuis l'interieur."""
    basse, haute = (sp.sympify(b) for b in bornes)
    en_haut = sp.limit(primitive, X, haute, "-")
    en_bas = sp.limit(primitive, X, basse, "+")
    valeur = sp.simplify(en_haut - en_bas)
    return Definie(integrande, (basse, haute), methode, primitive, valeur)


def impaire(integrande: sp.Expr, borne: sp.Expr) -> Definie:
    """Un integrande impair sur `[-b, b]` : les deux moities se compensent."""
    if sp.simplify(integrande.subs(X, -X) + integrande) != 0:
        msg = f"integrande non impair : {integrande}"
        raise ValueError(msg)
    methode = "integrande impair sur un intervalle symetrique"
    return Definie(integrande, (-borne, borne), methode, None, sp.Integer(0))


def par_symetrie(integrande: sp.Expr, autre: Definie, facteur: sp.Expr) -> Definie:
    """Une integrale ramenee a `facteur` fois une autre, de memes bornes."""
    methode = f"ramenee a une autre integrale ({autre.methode})"
    valeur = sp.simplify(facteur * autre.valeur)
    return Definie(integrande, autre.bornes, methode, None, valeur)


def racine(valeur: sp.Expr) -> sp.Expr:
    """`racine(valeur)`, abrege pour les integrandes a radicaux."""
    return sp.sqrt(valeur)
