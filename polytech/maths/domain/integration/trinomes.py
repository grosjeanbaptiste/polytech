"""Les radicaux et les inverses d'un trinome du second degre.

`Formulaire_Aide_memoire.pdf`, 5.4 et 5.7 (pages 6 et 7) :

* regle 3 -- pour primitiver `racine(ax^2 + bx + c)`, se ramener a
  `racine(u^2 +- p^2)` ou `racine(p^2 - u^2)` en completant le carre ;
* regle 4 -- pour `(Ax + B) / racine(ax^2 + bx + c)`, faire apparaitre au
  numerateur la derivee `2ax + b` du trinome, puis une constante.

Le formulaire ecrit la regle 4 sans le coefficient du premier terme : la
decomposition exacte est `Ax + B = (A / 2a)(2ax + b) + K`.

Les « cas particuliers » du 5.4 sont une donnee, recopiee de la page ; un
test redérive chaque ligne.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.symboles import VARIABLE as X

PARAMETRE: Final = sp.Symbol("p", positive=True)
"""Le `p` des cas particuliers."""

_PLUS, _MOINS = sp.sqrt(X**2 + PARAMETRE**2), sp.sqrt(X**2 - PARAMETRE**2)
_BAS = sp.sqrt(PARAMETRE**2 - X**2)

CAS_PARTICULIERS: Final[dict[sp.Expr, sp.Expr]] = {
    _PLUS: X / 2 * _PLUS + PARAMETRE**2 / 2 * sp.log(sp.Abs((X + _PLUS) / PARAMETRE)),
    _MOINS: X / 2 * _MOINS
    - PARAMETRE**2 / 2 * sp.log(sp.Abs((X + _MOINS) / PARAMETRE)),
    _BAS: PARAMETRE**2 / 2 * sp.asin(X / PARAMETRE) + X / 2 * _BAS,
    1 / sp.sqrt(X**2 - 1): sp.log(sp.Abs(X + sp.sqrt(X**2 - 1))),
    1 / (1 + X**2) ** 2: (sp.atan(X) + X / (1 + X**2)) / 2,
    1 / (1 + X**2) ** 3: sp.Rational(3, 8) * (sp.atan(X) + X / (1 + X**2))
    + X / (4 * (1 + X**2) ** 2),
}
"""L'integrande et sa primitive ; la premiere ligne du formulaire en vaut deux."""


def completer_le_carre(trinome: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`(a, h, k)` tels que `ax^2 + bx + c = a (x - h)^2 + k`."""
    polynome = sp.Poly(sp.expand(trinome), X)
    if polynome.degree() != 2:  # noqa: PLR2004 -- un trinome est du second degre
        msg = f"{trinome} n'est pas un trinome du second degre"
        raise ValueError(msg)
    a, b, c = polynome.all_coeffs()
    return a, -b / (2 * a), c - b**2 / (4 * a)


def decomposer_le_numerateur(
    numerateur: sp.Expr, trinome: sp.Expr
) -> tuple[sp.Expr, sp.Expr]:
    """`(alpha, K)` tels que `Ax + B = alpha (2ax + b) + K`."""
    derivee = sp.diff(sp.expand(trinome), X)
    alpha = sp.Poly(numerateur, X).coeff_monomial(X) / derivee.coeff(X, 1)
    return alpha, sp.expand(numerateur - alpha * derivee)


def _reduit(trinome: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`(a, u, k / a)` : le trinome vaut `a (u^2 + k / a)`."""
    a, h, k = completer_le_carre(trinome)
    if a.is_negative and not k.is_positive:
        msg = f"{trinome} n'est jamais positif : sa racine n'existe nulle part"
        raise ValueError(msg)
    return a, X - h, k / a


def primitive_de_la_racine(trinome: sp.Expr) -> sp.Expr:
    """Regle 3 : la primitive de `racine(ax^2 + bx + c)`."""
    a, u, rapport = _reduit(trinome)
    if rapport == 0:
        msg = f"{trinome} est un carre parfait : sa racine est une valeur absolue"
        raise ValueError(msg)
    modele = _BAS if a.is_negative else (_PLUS if rapport.is_positive else _MOINS)
    valeurs = {X: u, PARAMETRE: sp.sqrt(sp.Abs(rapport))}
    return sp.sqrt(sp.Abs(a)) * CAS_PARTICULIERS[modele].subs(
        valeurs, simultaneous=True
    )


def primitive_sur_la_racine(numerateur: sp.Expr, trinome: sp.Expr) -> sp.Expr:
    """Regle 4 : la primitive de `(Ax + B) / racine(ax^2 + bx + c)`."""
    alpha, constante = decomposer_le_numerateur(numerateur, trinome)
    a, u, rapport = _reduit(trinome)
    if a.is_negative:
        inverse = sp.asin(u / sp.sqrt(-rapport))
    else:
        inverse = sp.log(sp.Abs(u + sp.sqrt(u**2 + rapport)))
    return 2 * alpha * sp.sqrt(trinome) + constante * inverse / sp.sqrt(sp.Abs(a))


def primitive_de_l_inverse(trinome: sp.Expr) -> sp.Expr:
    """La primitive de `1 / (ax^2 + bx + c)` quand le trinome n'a pas de racine."""
    a, h, k = completer_le_carre(trinome)
    if not (k / a).is_positive:
        msg = f"{trinome} a des racines reelles : le decomposer en fractions simples"
        raise ValueError(msg)
    largeur = sp.sqrt(k / a)
    return sp.atan((X - h) / largeur) / (a * largeur)
