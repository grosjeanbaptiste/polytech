"""Stabilite et reponse indicielle, lues sur la fonction de transfert.

Le corrige decompose en fractions simples a la main. La decomposition n'est
pas le resultat : le resultat est la fonction du temps. Ces deux fonctions
la produisent sans passer par les residus, ce qui donne au test un chemin
different de celui du corrige -- condition pour qu'il puisse le contredire.
"""

from __future__ import annotations

import sympy as sp


def _denominateur(transfert: sp.Expr, variable: sp.Expr) -> sp.Poly:
    """Le denominateur de la fonction de transfert, en polynome."""
    _, denominateur = sp.fraction(sp.together(transfert))
    polynome = sp.Poly(sp.expand(denominateur), variable)
    if polynome.degree() < 1:
        msg = f"denominateur sans pole : {denominateur}"
        raise ValueError(msg)
    return polynome


def est_stable(transfert: sp.Expr, variable: sp.Expr) -> bool:
    """Vrai si tous les poles sont a partie reelle strictement negative."""
    racines = sp.roots(_denominateur(transfert, variable))
    if not racines:
        msg = "poles non calculables : la stabilite ne se decide pas"
        raise ValueError(msg)
    return all(bool(sp.re(racine) < 0) for racine in racines)


def reponse_a_un_echelon(
    transfert: sp.Expr,
    amplitude: sp.Expr,
    variable: sp.Expr,
    temps: sp.Expr,
) -> sp.Expr:
    """La sortie temporelle pour une entree `amplitude * u(t)`.

    L'echelon vaut `amplitude / s` en Laplace ; l'echelon d'Heaviside est
    ensuite ramene a 1, l'expression ne valant que pour `t > 0`.
    """
    image = sp.simplify(transfert * amplitude / variable)
    reponse = sp.inverse_laplace_transform(image, variable, temps)
    return sp.simplify(reponse.subs(sp.Heaviside(temps), 1))
