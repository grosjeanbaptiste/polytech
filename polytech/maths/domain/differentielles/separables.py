"""Les equations a variables separees, `y' = f(x) g(y)` (planches 22-23).

On separe, `dy / g(y) = f(x) dx`, puis on integre chaque membre.
« Attention aux constantes » : diviser par `g(y)` suppose `g(y) != 0`, et
les zeros de `g` sont des solutions **constantes** que la division perd.
Sur `y' = -2xy` la planche retrouve ainsi `y = 0`, ce qui permet a la
constante de la solution generale de parcourir tous les reels.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import (
    FONCTION,
    INCONNUE,
    VARIABLE,
)
from polytech.maths.domain.symboles import CONSTANTE
from polytech.raisonnement.domain.etapes import Etape, Raisonnement

SOURCE: Final = (
    "Mathematique/SLIDES/Slides_02_Equations_differentielles.pdf, planches 22-23"
)
PREMIERE_CONSTANTE: Final = sp.Symbol("K_1", real=True)


def solutions_constantes(en_y: sp.Expr) -> tuple[sp.Expr, ...]:
    """Les zeros reels de `g` : autant de solutions `y = c` de `y' = f(x) g(y)`."""
    racines = sp.solveset(en_y, INCONNUE, domain=sp.S.Reals)
    return tuple(sorted(racines, key=sp.default_sort_key))


def _explicite(en_x: sp.Expr, en_y: sp.Expr) -> sp.Expr:
    inconnue = FONCTION(VARIABLE)
    equation = sp.Eq(inconnue.diff(VARIABLE), en_x * en_y.subs(INCONNUE, inconnue))
    resolue = sp.dsolve(equation, inconnue)
    premiere = resolue[0] if isinstance(resolue, list) else resolue
    return premiere.rhs.subs(sp.Symbol("C1"), CONSTANTE)


def _est_dans_la_famille(constante: sp.Expr, famille: sp.Expr) -> bool:
    """Vrai si une valeur finie de `C` rend la solution constante."""
    valeurs = sp.solve(sp.Eq(famille, constante), CONSTANTE)
    return any(not v.has(VARIABLE) and v.is_finite for v in valeurs)


def resoudre_par_separation(en_x: sp.Expr, en_y: sp.Expr) -> Raisonnement:
    """Les etapes de la planche 23 pour `y' = f(x) g(y)`."""
    membre = "dy/y" if en_y == INCONNUE else f"dy/({en_y})"
    primitive_en_y = sp.integrate(1 / en_y, INCONNUE)
    affichee = primitive_en_y.replace(sp.log, lambda a: sp.log(sp.Abs(a)))
    primitive_en_x = sp.integrate(en_x, VARIABLE)
    famille = _explicite(en_x, en_y)
    etapes = [
        Etape(f"{membre} = {en_x} dx", "on separe les variables", SOURCE),
        Etape(
            f"{affichee} = {primitive_en_x} + {PREMIERE_CONSTANTE}",
            "on integre chaque membre ; une seule constante suffit",
            SOURCE,
        ),
        Etape(f"y = {famille}", "on isole y", SOURCE),
    ]
    constantes = solutions_constantes(en_y)
    etapes += [
        Etape(
            f"y = {c} est aussi solution",
            f"g({c}) = 0 : la division par g(y) l'avait ecartee",
            SOURCE,
        )
        for c in constantes
    ]
    retrouvees = constantes and all(
        _est_dans_la_famille(c, famille) for c in constantes
    )
    suffixe = ", C reel quelconque" if retrouvees else ""
    return Raisonnement(
        "Quelles sont les solutions de y' = f(x) g(y) ?",
        tuple(etapes),
        f"y = {famille}{suffixe}",
    )
