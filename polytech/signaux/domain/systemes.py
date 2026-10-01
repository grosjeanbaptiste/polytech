"""Les trois proprietes qui font qu'un systeme est justiciable du cours.

Lineaire, invariant, causal. Le cours ne les enonce pas pour le plaisir :
c'est leur conjonction qui autorise la convolution, donc la transmittance,
donc tout le reste. Un systeme qui en manque une sort du cadre, et le
corrige doit le dire avant de calculer quoi que ce soit.

Les proprietes sont testees sur des signaux symboliques, et non sur des
echantillons : deux signaux quelconques `x1` et `x2`, et un retard
quelconque. C'est une verification, pas un sondage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import TEMPS

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class Systeme:
    """Un systeme donne par l'operateur qui transforme son entree."""

    operateur: Callable[[sp.Expr], sp.Expr]

    def est_lineaire(self) -> bool:
        """`S(a x1 + b x2) = a S(x1) + b S(x2)` pour des signaux quelconques."""
        premier, second = sp.Function("x_1"), sp.Function("x_2")
        a, b = sp.symbols("a b")
        combinee = a * premier(TEMPS) + b * second(TEMPS)
        ecart = self.operateur(combinee) - (
            a * self.operateur(premier(TEMPS)) + b * self.operateur(second(TEMPS))
        )
        return bool(sp.simplify(sp.expand(ecart)) == 0)

    def est_invariant(self) -> bool:
        """Retarder l'entree doit retarder la sortie d'autant."""
        entree = sp.Function("x")
        retard = sp.Symbol("tau", positive=True)
        retardee = self.operateur(entree(TEMPS - retard))
        attendue = self.operateur(entree(TEMPS)).subs(TEMPS, TEMPS - retard)
        return bool(sp.simplify(retardee - attendue) == 0)


def convolution(premier: sp.Expr, second: sp.Expr) -> sp.Expr:
    """Le produit de convolution de deux signaux causaux.

    Sur des signaux causaux l'integrale court de zero a `t`, et non de
    moins l'infini a plus l'infini : c'est la causalite qui raccourcit les
    bornes, et l'ecrire ici evite de la reinventer a chaque exercice.
    """
    variable = sp.Symbol("theta", nonnegative=True)
    integrande = premier.subs(TEMPS, variable) * second.subs(TEMPS, TEMPS - variable)
    return sp.simplify(sp.integrate(integrande, (variable, 0, TEMPS)))


def linearisation_de_taylor(
    modele: sp.Expr,
    variable: sp.Expr,
    point: sp.Expr,
) -> sp.Expr:
    """`f(x_0) + f'(x_0) (x - x_0)` : le developpement limite au premier terme.

    C'est la seule facon de ramener un systeme physique au cadre LTI du
    cours, et le fascicule le dit ainsi. Ce qu'elle coute est dit aussi :
    le modele obtenu ne vaut qu'au **voisinage** du point, et c'est
    pourquoi on parle de modele local par opposition au modele global.
    """
    valeur = modele.subs(variable, point)
    pente = sp.diff(modele, variable).subs(variable, point)
    return sp.expand(valeur + pente * (variable - point))


def point_de_fonctionnement(modele: sp.Expr, variable: sp.Expr) -> tuple[sp.Expr, ...]:
    """Les valeurs ou le modele s'annule : les equilibres du systeme.

    Ce sont les points autour desquels on linearise. Un systeme sans
    equilibre n'a pas de modele local, et le dire vaut mieux que de rendre
    une liste vide qu'un appelant oublierait de regarder.
    """
    racines = sp.solve(sp.Eq(modele, 0), variable)
    if not racines:
        msg = f"aucun point de fonctionnement : {modele} ne s'annule jamais"
        raise ValueError(msg)
    return tuple(sorted(racines, key=sp.default_sort_key))


def est_causal(systeme: Systeme) -> bool:
    """Vrai si la sortie ne depend pas de l'entree future.

    Le test applique au systeme une entree **avancee** et verifie que la
    sortie l'est aussi -- un systeme qui anticiperait rendrait une sortie
    dependant de `t + tau` sans qu'on le lui ait demande. Hors du cadre du
    cours, et a dire avant de lui chercher une transmittance.
    """
    entree = sp.Function("x")
    avance = sp.Symbol("tau", positive=True)
    sortie = systeme.operateur(entree(TEMPS))
    return not sortie.has(entree(TEMPS + avance))
