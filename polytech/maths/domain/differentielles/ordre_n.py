"""La voie classique pour `K(D) y = x(t)` : SGENH = SGEH + SPENH.

Mathematiques appliquees, 1.4.2 : la solution generale de l'equation non
homogene est celle de l'equation homogene -- une combinaison de la base,
avec autant de constantes que l'ordre -- plus une solution particuliere.

**Superposition** (1.4.6) : `K(D)(y_1 + y_2) = x_1 + x_2`. Une
sollicitation qui est une somme se traite terme a terme. Chaque terme est
ici ecrit `P(t) e^(lambda t)`, `lambda` complexe : un cosinus ou un sinus
est une somme de deux exponentielles conjuguees, et `y = e^(lambda t) z`
ramene au cas d'un polynome (`particulieres.pour_un_polynome_module`).

Les **constantes** se fixent ensuite par les conditions initiales
`y(0), y'(0), ...` : « un systeme lineaire de n equations a n inconnues »
(3.6, exemple 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.domain.differentielles.particulieres import (
    pour_un_polynome_module,
)
from polytech.maths.domain.symboles import TEMPS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.maths.domain.differentielles.operateurs import OperateurDifferentiel


def constantes(nombre: int) -> tuple[sp.Symbol, ...]:
    """`c1, ..., cn`, les constantes de la solution generale."""
    return tuple(sp.symbols(f"c1:{nombre + 1}"))


def _polynome_et_taux(terme: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`(P, lambda)` tels que `terme = P(t) e^(lambda t)` ; leve sinon."""
    exposant, polynome = sp.Integer(0), sp.Integer(1)
    for facteur in sp.Mul.make_args(terme):
        if isinstance(facteur, sp.exp):
            exposant += facteur.args[0]
        else:
            polynome *= facteur
    taux = sp.simplify(exposant / TEMPS)
    if taux.has(TEMPS) or not polynome.is_polynomial(TEMPS):
        msg = f"sollicitation ni polynome, ni exponentielle, ni sinusoide : {terme}"
        raise ValueError(msg)
    return polynome, taux


def particuliere_par_superposition(
    operateur: OperateurDifferentiel, sollicitation: object
) -> sp.Expr:
    """Une solution particuliere, somme des reponses a chaque terme."""
    exponentielle = sp.expand(sp.sympify(sollicitation).rewrite(sp.exp))
    reponses = (
        pour_un_polynome_module(operateur, *_polynome_et_taux(terme))
        for terme in sp.Add.make_args(exponentielle)
        if terme != 0
    )
    somme = sum(reponses, sp.Integer(0))
    return sp.simplify(sp.expand(somme, complex=True))


def solution_generale(
    operateur: OperateurDifferentiel, sollicitation: object = 0
) -> sp.Expr:
    """SGEH + SPENH : la base ponderee par `c1, ..., cn`, plus une particuliere."""
    base = operateur.base_des_solutions()
    homogene = sum(
        (c * y for c, y in zip(constantes(len(base)), base, strict=True)),
        sp.Integer(0),
    )
    return homogene + particuliere_par_superposition(operateur, sollicitation)


def solution_de_cauchy(
    operateur: OperateurDifferentiel, sollicitation: object, initiales: Sequence[object]
) -> sp.Expr:
    """La solution qui prend en 0 les valeurs `y(0), y'(0), ...` donnees."""
    if len(initiales) != operateur.ordre:
        msg = (
            f"{operateur.ordre} conditions initiales attendues, {len(initiales)} recues"
        )
        raise ValueError(msg)
    generale = solution_generale(operateur, sollicitation)
    equations = [
        sp.diff(generale, TEMPS, k).subs(TEMPS, 0) - sp.sympify(valeur)
        for k, valeur in enumerate(initiales)
    ]
    (valeurs,) = sp.linsolve(equations, constantes(operateur.ordre))
    remplacement = dict(zip(constantes(operateur.ordre), valeurs, strict=True))
    return sp.simplify(generale.subs(remplacement))
