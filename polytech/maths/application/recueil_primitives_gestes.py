"""Les gestes du recueil de primitives, chacun avec la garde qui le tient.

Une resolution montre une reecriture ou un changement de variable. Si la
reecriture ne vaut pas l'integrande, ou si le changement ne le redonne pas,
le corrige afficherait une etape fausse au-dessus d'un resultat juste :
chaque geste le verifie donc avant de rendre, sur des points de controle
et non par simplification -- `racine(1 - cos x) = racine(2) sin(x/2)` n'est
vrai que sur un intervalle, et une simplification formelle le refuserait.
"""

from __future__ import annotations

from typing import Final

import sympy as sp

from polytech.maths.application import recueil_primitives_enonces as enonces
from polytech.maths.application.resolution_exs_01 import Resolution
from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.integration.substitution import NOUVELLE as U
from polytech.maths.domain.symboles import VARIABLE as X

__all__ = ["U", "changement", "changement_inverse", "ecart", "immediate"]
__all__ += ["T", "en_cascade", "par_parties", "terme_a_terme", "valeur_absolue"]

T: Final = sp.Symbol("t", positive=True)
"""La variable des changements `x = phi(t)`, qui font disparaitre un radical."""
POINTS: Final = (sp.Rational(1, 5), sp.Rational(7, 20), sp.Rational(1, 2))
"""Dans le domaine de tous les integrandes du recueil : `0 < x < 1/racine(3)`."""
VALEURS: Final = {
    enonces.A: sp.Rational(13, 10),
    enonces.B: sp.Rational(7, 10),
    enonces.P: sp.Rational(17, 10),
    enonces.N: 3,
}
PAR_PARTIES: Final = "integration par parties"
SEUIL: Final = 1e-12
CHIFFRES: Final = 30


def ecart(gauche: sp.Expr, droite: sp.Expr) -> float:
    """Le plus grand ecart relatif entre deux expressions aux points de controle."""
    difference, echelle = (gauche - droite).subs(VALEURS), droite.subs(VALEURS)
    ecarts = [
        abs(sp.N(difference.subs(X, point), CHIFFRES))
        / (1 + abs(sp.N(echelle.subs(X, point), CHIFFRES)))
        for point in POINTS
    ]
    return float(max(ecarts))


def valeur_absolue(expression: sp.Expr) -> sp.Expr:
    """`ln u` devient `ln |u|` : la primitive de `1/u` vaut des deux cotes de zero."""
    return expression.replace(sp.log, lambda argument: sp.log(sp.Abs(argument)))


def terme_a_terme(integrande: sp.Expr, reecrit: sp.Expr, methode: str) -> Resolution:
    """L'integrande reecrit en une somme, primitivee terme a terme.

    Terme a terme, et non d'un bloc : pour `(1 + tg^2 x) - 1` garde tel
    quel, `sympy` rend d'un bloc `tg x - 2x`. Le test du recueil l'a vu.
    """
    if ecart(reecrit, integrande) > SEUIL:
        msg = f"la reecriture {reecrit} ne vaut pas l'integrande {integrande}"
        raise ValueError(msg)
    termes = sp.Add.make_args(reecrit)
    primitive = sp.Add(*[sp.integrate(terme, X) for terme in termes])
    return Resolution(integrande, methode, (reecrit,), primitive)


def changement(integrande: sp.Expr, en_u: sp.Expr, u: sp.Expr) -> Resolution:
    """`u = g(x)` : l'integrande en `u` est donne ecrit, et verifie."""
    if ecart(en_u.subs(U, u) * sp.diff(u, X), integrande) > SEUIL:
        msg = f"le changement u = {u} ne redonne pas l'integrande {integrande}"
        raise ValueError(msg)
    primitive_en_u = valeur_absolue(sp.integrate(en_u, U))
    etapes = (u, en_u, primitive_en_u)
    return Resolution(
        integrande, "changement de variable", etapes, primitive_en_u.subs(U, u)
    )


def changement_inverse(
    integrande: sp.Expr, x_de_t: sp.Expr, t_de_x: sp.Expr, en_t: sp.Expr
) -> Resolution:
    """`x = phi(t)` : l'integrande en `t` est donne ecrit, et verifie."""
    transporte = (en_t / sp.diff(x_de_t, T)).subs(T, t_de_x)
    if ecart(transporte, integrande) > SEUIL:
        msg = f"le changement x = {x_de_t} ne redonne pas l'integrande {integrande}"
        raise ValueError(msg)
    primitive_en_t = valeur_absolue(sp.integrate(en_t, T))
    etapes = (x_de_t, en_t, primitive_en_t)
    return Resolution(
        integrande, "changement de variable", etapes, primitive_en_t.subs(T, t_de_x)
    )


def immediate(integrande: sp.Expr, primitive: sp.Expr) -> Resolution:
    """Une primitive que la table donne : on la verifie en la derivant."""
    if ecart(sp.diff(primitive, X), integrande) > SEUIL:
        msg = f"{primitive} n'est pas une primitive de {integrande}"
        raise ValueError(msg)
    return Resolution(integrande, "primitive immediate", (primitive,), primitive)


def _etape(u: sp.Expr, dv: sp.Expr) -> sp.Expr:
    """`u v - integrale de v du`, l'integrale qui reste etant simplifiee."""
    geste = IntegrationParParties(u, dv)
    restante = sp.simplify(geste.v() * geste.derivee_de_u())
    return geste.u * geste.v() - sp.Integral(restante, X)


def par_parties(u: sp.Expr, dv: sp.Expr) -> Resolution:
    """Une integration par parties, l'integrale restante se calculant directement."""
    etape = _etape(u, dv)
    return Resolution(u * dv, PAR_PARTIES, (etape,), etape.doit())


def en_cascade(us: tuple[sp.Expr, ...], dv: sp.Expr) -> Resolution:
    """Des parties en chaine : chaque `u` est celui de l'integrale qui restait."""
    etapes = tuple(_etape(u, dv) for u in us)
    primitive = sp.expand(sp.integrate(us[0] * dv, X))
    return Resolution(us[0] * dv, PAR_PARTIES, etapes, primitive)
