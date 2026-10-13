"""Ecrire une primitive comme le cours l'ecrirait, sans en changer la valeur.

`sympy` rend une primitive juste dans une ecriture qui n'est pas toujours
celle d'un corrige : `sin x / cos x` pour `tg x`, un dephasage de `pi/4`
la ou l'on attend un sinus et un cosinus, un polynome developpe la ou un
facteur se lit mieux. Ce module ne fait que ranger ; le test verifie que
la derivee n'a pas bouge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable


def _en_sinus_et_cosinus(expression: sp.Expr) -> sp.Expr:
    """Developpe les dephasages, puis regroupe par fonction trigonometrique."""
    developpee = sp.expand(sp.expand_trig(expression))
    fonctions = sorted(
        developpee.atoms(sp.sin, sp.cos, sp.sinh, sp.cosh), key=sp.default_sort_key
    )
    return sp.collect(developpee, fonctions)


def _en_facteur(expression: sp.Expr) -> sp.Expr:
    return sp.factor(expression)


def _en_tangente(expression: sp.Expr) -> sp.Expr:
    return sp.simplify(expression)


def _sous_l_exponentielle(expression: sp.Expr) -> sp.Expr:
    """Met l'exponentielle et le denominateur en facteur, puis regroupe le reste."""
    exponentielle = next(iter(expression.atoms(sp.exp)))
    reste = sp.expand(sp.simplify(expression / exponentielle))
    denominateur = sp.denom(sp.together(reste))
    developpe = sp.expand(reste * denominateur)
    fonctions = sorted(developpe.atoms(sp.sin, sp.cos), key=sp.default_sort_key)
    interieur = sp.collect(developpe, fonctions)
    return sp.Mul(exponentielle / denominateur, interieur, evaluate=False)


RANGEMENTS: Final[dict[int, Callable[[sp.Expr], sp.Expr]]] = {
    4: _en_facteur,
    12: _en_tangente,
    13: _en_facteur,
    26: _en_sinus_et_cosinus,
    37: _en_sinus_et_cosinus,
    40: _sous_l_exponentielle,
    42: _en_sinus_et_cosinus,
    47: _en_tangente,
    52: _en_sinus_et_cosinus,
}
"""Numero de la primitive de l'exercice 1 : le rangement que son ecriture demande."""


def presenter(numero: int, primitive: sp.Expr) -> sp.Expr:
    """La primitive, rangee si son numero le demande ; inchangee sinon."""
    ranger = RANGEMENTS.get(numero)
    return ranger(primitive) if ranger else primitive
