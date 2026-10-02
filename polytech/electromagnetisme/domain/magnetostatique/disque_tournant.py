"""Le disque de Faraday -- exercice 12.

Un disque conducteur de diametre `D` tourne a la vitesse angulaire
`omega` dans un champ `B` perpendiculaire. On recueille une tension entre
son centre et son bord : c'est le premier generateur electrique de
l'histoire, et le plus simple qui soit.

**Trois routes menent a la meme tension**, et elles ne partagent rien.

`tension_induite` integre la force de Laplace le long du rayon -- la
route du manuscrit.

`tension_par_le_flux_balaye` multiplie `B` par l'aire que le rayon
balaie par unite de temps. Aucune integrale de chemin, aucun produit
vectoriel.

`tension_par_la_tige_equivalente` traite le disque comme une tige de
longueur `D/2` se deplacant tout entiere a la vitesse du point situe a
mi-rayon. Elle ne tient que parce que l'integrande est **lineaire** en
`r` : la moyenne d'une fonction lineaire est sa valeur au milieu. Un
champ non uniforme la romprait.

**Le paradoxe du disque.** Le flux a travers un contour geometrique fixe
ne varie pas -- `B` est uniforme et la geometrie immobile -- de sorte que
`e = -dPhi/dt` applique a ce contour donnerait zero. C'est faux. La
raison est que dans un conducteur en rotation le « circuit » n'est pas
defini : les porteurs changent de place a chaque instant. C'est le flux
**balaye** qui compte, non le flux enlace.
"""

from __future__ import annotations

import sympy as sp


def produit_vectoriel_cylindrique(
    premier: sp.Matrix,
    second: sp.Matrix,
) -> sp.Matrix:
    """Le produit vectoriel, dans la base cylindrique directe.

    La base `(u_r, u_phi, u_z)` est directe, donc le produit vectoriel
    s'y calcule comme en cartesiennes -- ce qui permet de **verifier**
    `u_phi ^ u_z = u_r` plutot que de citer l'identite.
    """
    return sp.Matrix(premier).cross(sp.Matrix(second))


def vitesse_lineaire(pulsation: sp.Expr, rayon: sp.Expr) -> sp.Expr:
    """`v = omega r` : la seule grandeur du calcul qui depende de `r`.

    Sa linearite est ce qui rend l'integrande lineaire, donc
    l'equivalence a la tige du rayon moyen possible.
    """
    return pulsation * rayon


def tension_induite(
    induction: sp.Expr,
    pulsation: sp.Expr,
    diametre: sp.Expr,
) -> sp.Expr:
    """`e = integrale de B omega r dr` du centre au bord : `B omega D^2 / 8`.

    Le produit vectoriel `v ^ B` est **radial** -- `u_phi ^ u_z = u_r` --
    de sorte que le produit scalaire avec `dr u_r` ne perd rien. Une
    composante azimutale ou axiale ne contribuerait pas au chemin.
    """
    rayon = sp.Symbol("r_integration", positive=True)
    integrande = induction * vitesse_lineaire(pulsation, rayon)
    return sp.integrate(integrande, (rayon, 0, diametre / 2))


def courant_debite(
    induction: sp.Expr,
    pulsation: sp.Expr,
    diametre: sp.Expr,
    resistance: sp.Expr,
) -> sp.Expr:
    """`I = e / R` : la loi d'Ohm sur le circuit extérieur."""
    return tension_induite(induction, pulsation, diametre) / resistance


def tension_par_le_flux_balaye(
    induction: sp.Expr,
    pulsation: sp.Expr,
    diametre: sp.Expr,
) -> sp.Expr:
    """`e = B x (aire balayee par unite de temps)`.

    Le rayon balaie un secteur d'angle `omega` par unite de temps, donc
    une aire `omega (D/2)^2 / 2`. Route independante de la precedente :
    ni integrale de chemin, ni produit vectoriel.

    C'est la lecture **correcte** de la loi de Faraday pour cette
    configuration -- le flux balaye, non le flux enlace a travers un
    contour fixe, qui lui ne varie pas.
    """
    rayon_du_disque = diametre / 2
    aire_balayee = pulsation * rayon_du_disque**2 / 2
    return induction * aire_balayee


def vitesse_au_rayon_moyen(pulsation: sp.Expr, diametre: sp.Expr) -> sp.Expr:
    """`omega D / 4` : la vitesse a mi-rayon, moitie de celle du bord.

    Elle vaut aussi la **moyenne** des vitesses du centre au bord, et
    c'est cette coincidence -- propre aux fonctions lineaires -- qui
    fonde l'equivalence a la tige.
    """
    return vitesse_lineaire(pulsation, diametre / 4)


def tension_par_la_tige_equivalente(
    induction: sp.Expr,
    pulsation: sp.Expr,
    diametre: sp.Expr,
) -> sp.Expr:
    """`e = B v_moyen (D/2)` : le disque vu comme une tige mobile.

    Troisieme route, et la plus parlante. Elle ne tient que parce que
    l'integrande est lineaire ; avec un champ non uniforme, la valeur au
    milieu cesserait d'etre la moyenne et l'equivalence tomberait.
    """
    longueur = diametre / 2
    return induction * vitesse_au_rayon_moyen(pulsation, diametre) * longueur
