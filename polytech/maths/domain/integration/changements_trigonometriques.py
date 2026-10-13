"""Faire disparaitre `racine(a^2 x^2 +- b^2)` par un changement trigonometrique.

`Formulaire_Aide_memoire.pdf`, 5.7 (page 7), regles 5 a 7 :

* `racine(a^2 x^2 + b^2)` -- poser `x = (b / a) tg t`, le radical devient
  `b / cos t` ;
* `racine(a^2 x^2 - b^2)` -- poser `x = (b / a) / cos t`, le radical
  devient `b tg t` ;
* `racine(b^2 - a^2 x^2)` -- poser `x = (b / a) sin t`, le radical devient
  `b cos t`.

Chaque fonction rend l'integrande en `t`, le facteur `dx` compris. Les
trois valeurs du radical supposent `t` dans le premier quadrant, ou
`cos t`, `sin t` et `tg t` sont positifs : c'est le choix du cours, qui
revient a `x` par `t = arccos(2 / x)` dans son exemple.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE as X

ANGLE: Final = sp.Symbol("t", real=True)
"""La nouvelle variable `t`."""


def _changer(
    integrande: sp.Expr, radical: sp.Expr, devient: sp.Expr, x_de_t: sp.Expr
) -> sp.Expr:
    en_t = sp.simplify(
        integrande.subs(radical, devient).subs(X, x_de_t) * sp.diff(x_de_t, ANGLE)
    )
    if any(p.exp.is_Rational and not p.exp.is_Integer for p in en_t.atoms(sp.Pow)):
        msg = f"le changement laisse un radical : {en_t}"
        raise ValueError(msg)
    return en_t


def _rapport(a: object, b: object) -> tuple[sp.Expr, sp.Expr]:
    premier, second = sp.sympify(a), sp.sympify(b)
    if not (premier.is_positive and second.is_positive):
        msg = f"a et b doivent etre strictement positifs : {premier}, {second}"
        raise ValueError(msg)
    return premier, second


def par_la_tangente(integrande: sp.Expr, a: object, b: object) -> sp.Expr:
    """Regle 5 : `x = (b / a) tg t` pour `racine(a^2 x^2 + b^2)`."""
    pente, cote = _rapport(a, b)
    radical = sp.sqrt(pente**2 * X**2 + cote**2)
    return _changer(
        integrande, radical, cote / sp.cos(ANGLE), cote / pente * sp.tan(ANGLE)
    )


def par_le_cosinus(integrande: sp.Expr, a: object, b: object) -> sp.Expr:
    """Regle 6 : `x = (b / a) / cos t` pour `racine(a^2 x^2 - b^2)`."""
    pente, cote = _rapport(a, b)
    radical = sp.sqrt(pente**2 * X**2 - cote**2)
    return _changer(
        integrande, radical, cote * sp.tan(ANGLE), cote / pente / sp.cos(ANGLE)
    )


def par_le_sinus(integrande: sp.Expr, a: object, b: object) -> sp.Expr:
    """Regle 7 : `x = (b / a) sin t` pour `racine(b^2 - a^2 x^2)`."""
    pente, cote = _rapport(a, b)
    radical = sp.sqrt(cote**2 - pente**2 * X**2)
    return _changer(
        integrande, radical, cote * sp.cos(ANGLE), cote / pente * sp.sin(ANGLE)
    )
