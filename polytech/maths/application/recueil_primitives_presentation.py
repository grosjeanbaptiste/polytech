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

from polytech.maths.domain.symboles import VARIABLE as X

if TYPE_CHECKING:
    from collections.abc import Callable


def _en_sinus_et_cosinus(expression: sp.Expr) -> sp.Expr:
    """Developpe les dephasages, puis regroupe par fonction trigonometrique."""
    developpee = sp.expand(sp.expand_trig(expression))
    fonctions = sorted(
        developpee.atoms(sp.sin, sp.cos, sp.sinh, sp.cosh), key=sp.default_sort_key
    )
    return sp.collect(developpee, fonctions)


def _par_fonction(expression: sp.Expr) -> sp.Expr:
    """Regroupe par sinus et cosinus, sans toucher a leurs angles."""
    developpee = sp.expand(expression)
    fonctions = sorted(developpee.atoms(sp.sin, sp.cos), key=sp.default_sort_key)
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
    interieur = _par_fonction(reste * denominateur)
    return sp.Mul(exponentielle / denominateur, interieur, evaluate=False)


def _rationnel_regroupe(expression: sp.Expr) -> sp.Expr:
    """Reunit en une seule fraction les termes sans logarithme ni arc tangente."""
    termes = sp.Add.make_args(expression)
    rationnels = [t for t in termes if t.is_rational_function(X)]
    autres = [t for t in termes if not t.is_rational_function(X)]
    fraction = sp.factor(sp.together(sp.Add(*rationnels)))
    return sp.Add(fraction, *autres)


def _simplifie(expression: sp.Expr) -> sp.Expr:
    return sp.simplify(sp.expand(expression))


def _sur_les_positifs(expression: sp.Expr) -> sp.Expr:
    """Simplifie en supposant `x > 0`, puis rend la variable du recueil."""
    positif = sp.Symbol("x", positive=True)
    simple = sp.radsimp(sp.simplify(expression.subs(X, positif)))
    return sp.factor_terms(simple).subs(positif, X)


def _sans_arc_de_tangente(expression: sp.Expr) -> sp.Expr:
    """`arctan(tan x) = x` sur `]-pi/2, pi/2[`, ou l'on travaille."""
    return expression.subs(sp.atan(sp.tan(X)), X)


RANGEMENTS: Final[dict[int | str, Callable[[sp.Expr], sp.Expr]]] = {
    4: _en_facteur,
    12: _en_tangente,
    13: _en_facteur,
    26: _en_sinus_et_cosinus,
    37: _en_sinus_et_cosinus,
    40: _sous_l_exponentielle,
    42: _en_sinus_et_cosinus,
    47: _en_tangente,
    52: _en_sinus_et_cosinus,
    "2g": _rationnel_regroupe,
    "2h": _rationnel_regroupe,
    "5c": _par_fonction,
    "6g": _sur_les_positifs,
    "8m": _sans_arc_de_tangente,
    "10aPlus": _simplifie,
    "10aMoins": _simplifie,
    "10d": _sur_les_positifs,
    "11b": _rationnel_regroupe,
    "11c": _simplifie,
    "11d": _simplifie,
    "11i": _en_sinus_et_cosinus,
}
"""Numero (exercice 1) ou cle (exercices suivants) : le rangement de son ecriture."""


def presenter(numero: int | str, primitive: sp.Expr) -> sp.Expr:
    """La primitive, rangee si son numero le demande ; inchangee sinon."""
    ranger = RANGEMENTS.get(numero)
    return ranger(primitive) if ranger else primitive
