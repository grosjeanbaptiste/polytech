"""La conservation de la quantite de mouvement, et l'ecoulement de Couette.

Navier-Stokes pour un fluide newtonien incompressible :
`rho (du/dt + (u.grad) u) = -grad p + mu laplacien(u) + rho g`. Les
planches la posent, puis la resolvent sur l'ecoulement de **Couette
totalement developpe** -- et c'est cette resolution qui sert d'ancrage,
parce que son resultat est exact et verifiable : un profil **lineaire**,
et un cisaillement **uniforme**.

Le profil n'est pas suppose lineaire : il est obtenu en integrant
l'equation reduite, et le test verifie son degre.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.symboles import Y

VISCOSITE = sp.Symbol("mu", positive=True)
HAUTEUR = sp.Symbol("h", positive=True)
VITESSE_DE_PAROI = sp.Symbol("U", positive=True)
CONSTANTES = (sp.Symbol("A", real=True), sp.Symbol("B", real=True))


def equation_reduite() -> sp.Eq:
    """`d^2 u / dy^2 = 0` : ce que devient Navier-Stokes pour Couette.

    Totalement developpe, donc aucune variation selon `x` ; stationnaire,
    donc aucune derivee en temps ; sans gradient de pression impose, donc
    plus de second membre. Il ne reste que le laplacien.
    """
    inconnue = sp.Function("u")(Y)
    return sp.Eq(sp.Derivative(inconnue, Y, 2), 0)


def profil_general() -> sp.Expr:
    """`A y + B`, les deux integrations de l'equation reduite."""
    premiere, seconde = CONSTANTES
    return premiere * Y + seconde


def profil_de_couette() -> sp.Expr:
    """Le profil avec ses conditions : nul a la paroi fixe, `U` a la mobile."""
    conditions = [
        sp.Eq(profil_general().subs(Y, 0), 0),
        sp.Eq(profil_general().subs(Y, HAUTEUR), VITESSE_DE_PAROI),
    ]
    resolu = sp.solve(conditions, CONSTANTES, dict=True)
    if len(resolu) != 1:
        msg = f"conditions de Couette sans solution unique : {resolu}"
        raise ValueError(msg)
    return sp.simplify(profil_general().subs(resolu[0]))


def cisaillement() -> sp.Expr:
    """`tau = mu du/dy` : uniforme, et c'est la ce qui caracterise Couette."""
    return sp.simplify(VISCOSITE * sp.diff(profil_de_couette(), Y))
