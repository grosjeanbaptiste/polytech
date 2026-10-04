"""Revenir a l'original : les deux « regles de Heaviside ».

Quand `F(p)` n'est pas dans la table, deux procedes :

* **premiere regle** -- si `f` est developpable en serie de Mac Laurin,
  `L(t^n) = n! / p^(n+1)` donne `F(p) = f(0)/p + f'(0)/p^2 + ...` : on
  developpe `F` selon les puissances de `1/p`, et chaque `1/p^(n+1)`
  redevient `t^n / n!` ;
* **deuxieme regle** -- si `F = P/Q` est une fraction rationnelle ou le
  degre de `Q` depasse celui de `P`, on la decompose en fractions simples
  et l'on inverse terme a terme.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import IMAGE, TEMPS


def par_la_serie(image: sp.Expr, termes: int) -> sp.Expr:
    """Les `termes` premiers termes de la serie de Mac Laurin de l'original."""
    inverse = sp.Symbol("u", positive=True)
    developpement = sp.series(image.subs(IMAGE, 1 / inverse), inverse, 0, termes + 1)
    return sum(
        (
            developpement.coeff(inverse, n + 1) * TEMPS**n / sp.factorial(n)
            for n in range(termes)
        ),
        sp.Integer(0),
    )


def fractions_simples(image: sp.Expr) -> tuple[sp.Expr, ...]:
    """Les termes de la decomposition ; leve si la fraction n'est pas propre."""
    numerateur, denominateur = sp.fraction(sp.cancel(image))
    if sp.degree(numerateur, IMAGE) >= sp.degree(denominateur, IMAGE):
        msg = "degre du numerateur non inferieur a celui du denominateur"
        raise ValueError(msg)
    return tuple(sp.Add.make_args(sp.apart(image, IMAGE)))


def par_fractions_simples(image: sp.Expr) -> sp.Expr:
    """L'original pour `t >= 0`, somme des originaux de chaque fraction simple."""
    originaux = (
        sp.inverse_laplace_transform(terme, IMAGE, TEMPS, noconds=True)
        for terme in fractions_simples(image)
    )
    return sp.simplify(sum(originaux, sp.Integer(0)).subs(sp.Heaviside(TEMPS), 1))
