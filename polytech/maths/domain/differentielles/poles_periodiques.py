"""Poles de `tan` et `cot` sur un intervalle d'etude **borne**.

Les poles sont periodiques (`pi/2 + k pi` pour `tan`, `k pi` pour `cot`) :
leur liste est infinie sur `R`. Le contrat est donc de n'en rendre que ceux
de l'intervalle d'etude, que l'appelant fournit et qui doit etre borne.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.differentielles.premier_ordre import VARIABLE


def _poles_de(fonction: sp.Function, etude: sp.Interval) -> set[sp.Expr]:
    argument = fonction.args[0]
    pente = sp.diff(argument, VARIABLE)
    origine = argument.subs(VARIABLE, 0)
    if pente.free_symbols or sp.simplify(argument - pente * VARIABLE - origine) != 0:
        msg = f"argument non lineaire en {VARIABLE} : {argument}"
        raise ValueError(msg)
    decalage = sp.pi / 2 if isinstance(fonction, sp.tan) else sp.Integer(0)
    bornes = sorted(
        (pente * etude.inf + origine, pente * etude.sup + origine), key=float
    )
    bas = int(sp.floor((bornes[0] - decalage) / sp.pi)) - 1
    haut = int(sp.ceiling((bornes[1] - decalage) / sp.pi)) + 1
    candidats = ((decalage + k * sp.pi - origine) / pente for k in range(bas, haut + 1))
    return {x for x in candidats if etude.contains(x) == sp.true}


def poles_dans(expression: sp.Expr, etude: sp.Interval) -> sp.FiniteSet:
    """Les poles de `tan`/`cot` de l'expression qui tombent dans `etude`.

    Leve `ValueError` si `etude` n'est pas borne, ou si un argument n'est pas
    lineaire en la variable (les poles ne seraient plus periodiques).
    """
    if not (etude.inf.is_finite and etude.sup.is_finite):
        msg = f"intervalle d'etude non borne : {etude}"
        raise ValueError(msg)
    poles: set[sp.Expr] = set()
    for fonction in expression.atoms(sp.tan, sp.cot):
        poles |= _poles_de(fonction, etude)
    return sp.FiniteSet(*poles)
