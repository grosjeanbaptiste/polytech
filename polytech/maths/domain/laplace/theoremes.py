"""Le theoreme de la valeur finale (exercice II.3).

Si `f` et `f'` sont de type exponentiel **et si `lim f(t)` existe**, alors

    lim_{t -> +oo} f(t) = lim_{p -> 0+} p F(p).

La seconde hypothese n'est pas decorative : la limite de `p F(p)` se
calcule toujours, que celle de `f` existe ou non. Elle existe quand tous
les poles de `p F(p)` sont a partie reelle strictement negative ; un pole
a droite fait diverger `f`, une paire sur l'axe imaginaire la fait
osciller sans fin, et dans les deux cas la formule rend un nombre qui ne
dit rien de `f`.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.symboles import IMAGE


def limite_en_zero(image: sp.Expr) -> sp.Expr:
    """`lim_{p -> 0+} p F(p)` : ce que la formule rend, applicable ou non."""
    return sp.limit(IMAGE * image, IMAGE, 0, "+")


def poles_genants(image: sp.Expr) -> tuple[sp.Expr, ...]:
    """Les poles de `p F(p)` dont la partie reelle n'est pas negative."""
    _, denominateur = sp.fraction(sp.cancel(IMAGE * image))
    poles = sp.roots(sp.Poly(denominateur, IMAGE))
    return tuple(p for p in poles if not sp.re(p).is_negative)


def valeur_finale(image: sp.Expr) -> sp.Expr:
    """`lim f(t)`, par le theoreme ; leve si `f` n'a pas de limite."""
    genants = poles_genants(image)
    if genants:
        msg = f"theoreme inapplicable : f n'a pas de limite, pole(s) en {genants}"
        raise ValueError(msg)
    return limite_en_zero(image)
