"""L'induction au centre d'une nappe plane de spires -- seance 2.1, exercice 3.

Des spires jointives occupent une couronne plane, de `R_1` a `R_2`, et
portent ensemble le courant `I`. On la decoupe en **tubes de courant**
de largeur `dr`, chacun portant `dI = I dr / (R_2 - R_1)` : le courant
est reparti uniformement sur la largeur de la couronne.

Chaque tube est une spire de rayon `r`, dont l'induction au centre vaut
`mu_0 dI / (2 r)`. La somme est une integrale de `1/r` : d'ou le
logarithme, et le fait que les tubes proches du centre pesent davantage.

Le **signe** est celui de la figure du manuscrit, ou le courant tourne
de sorte que `ds ^ u_NP = -u_z`. Les fonctions rendent la composante
selon `u_z`, donc une valeur negative.
"""

from __future__ import annotations

import sympy as sp

RAYON = sp.Symbol("r_tube", positive=True)
"""Le rayon courant d'un tube, variable muette de l'integrale."""

AZIMUT = sp.Symbol("theta_tube", real=True)


def induction_d_une_spire_au_centre(
    courant: sp.Expr,
    rayon: sp.Expr,
    permeabilite: sp.Expr,
) -> sp.Expr:
    """`B_z = -mu_0 I / (2 R)`, par Biot et Savart integre sur un tour.

    Tous les elements de la spire sont a la meme distance `R` du centre
    et perpendiculaires au rayon : l'integrande est constant, et
    l'integrale se reduit a multiplier par la circonference.
    """
    element = permeabilite * courant / (4 * sp.pi * rayon**2) * rayon
    return -sp.integrate(element, (AZIMUT, 0, 2 * sp.pi))


def induction_de_la_nappe_au_centre(
    courant: sp.Expr,
    interieur: sp.Expr,
    exterieur: sp.Expr,
    permeabilite: sp.Expr,
) -> sp.Expr:
    """La somme des tubes, de `R_1` a `R_2` : composante selon `u_z`.

    Rendue sans resimplification : `sp.simplify` replie le logarithme en
    une puissance, forme exacte mais illisible, et que les
    developpements limites ne savent plus traiter.
    """
    densite = courant / (exterieur - interieur)
    tube = induction_d_une_spire_au_centre(densite, RAYON, permeabilite)
    return sp.integrate(tube, (RAYON, interieur, exterieur))


def moyenne_de_l_inverse_du_rayon(interieur: sp.Expr, exterieur: sp.Expr) -> sp.Expr:
    """`<1/r>` sur la couronne : ce que le logarithme du resultat mesure.

    L'induction vaut `mu_0 I / 2` fois cette moyenne -- la formule de la
    spire unique, ou `1/R` est remplace par sa moyenne sur la largeur.
    """
    somme = sp.integrate(1 / RAYON, (RAYON, interieur, exterieur))
    return sp.simplify(somme / (exterieur - interieur))
