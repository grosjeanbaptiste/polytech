"""La conservation de l'energie, et les deux applications des planches.

Bernoulli le long d'une ligne de courant :
`p + rho g z + rho V^2 / 2 = constante`. Les planches en tirent deux
applications, et toutes deux sont ici parce qu'elles servent d'ancrage :
le **reservoir de Torricelli**, `V = racine(2 g h)`, et le **tube de
Pitot**, `V = racine(2 Delta p / rho)`.

Les deux resultats ne sont pas tapes : ils sont **derives** de la
constante de Bernoulli par sympy, et les tests verifient qu'on retombe
bien sur les formules que le cours affiche.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.grandeurs import PESANTEUR

GRAVITE_EXACTE = sp.nsimplify(PESANTEUR, rational=True)
"""La pesanteur du cours, en rationnel exact plutot qu'en flottant.

`9.81` en flottant laisse un residu de l'ordre de `10^-15` dans toute
soustraction symbolique : la conservation de la charge, qui doit valoir
zero, vaudrait `3,55e-15 h rho`. Le rationnel l'annule exactement, et la
valeur n'est pas dupliquee -- elle est convertie depuis la constante du
cours.
"""

PRESSION = sp.Symbol("p", positive=True)
ALTITUDE = sp.Symbol("z", real=True)
VITESSE = sp.Symbol("V", nonnegative=True)
MASSE_VOLUMIQUE = sp.Symbol("rho", positive=True)


def charge(
    pression: sp.Expr = PRESSION,
    altitude: sp.Expr = ALTITUDE,
    vitesse: sp.Expr = VITESSE,
) -> sp.Expr:
    """`p + rho g z + rho V^2 / 2` : la quantite conservee."""
    return (
        pression
        + MASSE_VOLUMIQUE * GRAVITE_EXACTE * altitude
        + MASSE_VOLUMIQUE * vitesse**2 / 2
    )


def vitesse_de_torricelli(hauteur: sp.Expr) -> sp.Expr:
    """La vitesse de vidange, derivee de Bernoulli et non recopiee.

    Surface libre au repos a l'altitude `h`, orifice a l'altitude zero, et
    la meme pression atmospherique aux deux bouts : la charge se conserve,
    donc `rho g h = rho V^2 / 2`.
    """
    inconnue = sp.Symbol("V_s", nonnegative=True)
    en_haut = charge(pression=PRESSION, altitude=hauteur, vitesse=sp.Integer(0))
    en_bas = charge(pression=PRESSION, altitude=sp.Integer(0), vitesse=inconnue)
    racines = sp.solve(sp.Eq(en_haut, en_bas), inconnue)
    positives = [r for r in racines if not r.is_negative]
    return sp.simplify(positives[0])


def vitesse_de_pitot(ecart_de_pression: sp.Expr) -> sp.Expr:
    """La vitesse lue sur un Pitot, derivee du meme principe.

    Au point d'arret la vitesse est nulle et la pression vaut `p + rho V^2/2`
    -- l'ecart entre les deux prises est donc la pression dynamique.
    """
    inconnue = sp.Symbol("V_p", nonnegative=True)
    equation = sp.Eq(MASSE_VOLUMIQUE * inconnue**2 / 2, ecart_de_pression)
    racines = sp.solve(equation, inconnue)
    positives = [r for r in racines if not r.is_negative]
    return sp.simplify(positives[0])
