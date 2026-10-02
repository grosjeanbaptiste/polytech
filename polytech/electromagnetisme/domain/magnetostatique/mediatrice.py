"""Deux conducteurs opposes, vus de leur mediatrice -- exercice 6, seance 2.1.

Les conducteurs sont en `x = -a` et `x = +a`, paralleles a `Oz`, parcourus
par des courants **opposes**. Le point `P(0, b)` est sur la mediatrice,
donc a egale distance des deux : `sqrt(a^2 + b^2)`.

**Ce que l'egalite des distances fait.** Les contributions selon `Ox` sont
egales et opposees et se detruisent ; celles selon `Oy` s'ajoutent, d'ou
le facteur deux. Avec des courants de **meme sens**, c'est l'inverse qui
se produit -- et `champ_si_les_courants_sont_de_meme_sens` le rend
verifiable, parce que sans ce controle rien ne distinguerait une somme
d'une difference.

**Le champ est maximal pour `a = b`.** Pour une distance donnee, ecarter
les conducteurs augmente d'abord le champ puis le diminue : on gagne en
bras de levier, puis on perd en proximite. C'est le seul resultat de
l'exercice qui serve a dimensionner quelque chose, et le manuscrit n'en
parle pas.

**Au loin le champ decroit en `1/b^2`**, une puissance de plus qu'un fil
seul : les deux courants opposes forment un **dipole magnetique**. C'est
l'exact analogue de la charge et de son image a l'exercice 5 de statique.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.magnetostatique.superposition import Fil


def _distance_au_point(demi_ecart: sp.Expr, ordonnee: sp.Expr) -> sp.Expr:
    """`sqrt(a^2 + b^2)` : la meme pour les deux conducteurs.

    Commune parce que `P` est sur la mediatrice, et c'est cette egalite
    qui fait se detruire les composantes tangentielles.
    """
    return sp.sqrt(demi_ecart**2 + ordonnee**2)


def champ_sur_la_mediatrice(
    courant: sp.Expr,
    demi_ecart: sp.Expr,
    ordonnee: sp.Expr,
) -> tuple[sp.Expr, sp.Expr]:
    """Le champ `H` en `P(0, b)`, en composantes `(H_x, H_y)`.

    Rendu en deux composantes pour que l'annulation de la premiere soit
    **verifiable** plutot que supposee : c'est elle qui etablit que le
    champ est porte par `Oy`.

    Le cosinus de l'angle vaut `a / sqrt(a^2 + b^2)`, et c'est lui qui
    projette les deux contributions sur `Oy`.
    """
    distance = _distance_au_point(demi_ecart, ordonnee)
    module = Fil(courant=courant, distance=distance).module_du_champ()
    cosinus = demi_ecart / distance
    projection = module * cosinus
    # Les deux conducteurs portent des courants opposes : selon Ox leurs
    # contributions se detruisent, selon Oy elles s'ajoutent -- vers les
    # y negatifs, d'ou le signe.
    return projection - projection, -2 * projection


def champ_si_les_courants_sont_de_meme_sens(
    courant: sp.Expr,
    demi_ecart: sp.Expr,
    ordonnee: sp.Expr,
) -> tuple[sp.Expr, sp.Expr]:
    """Le meme calcul avec des courants de meme sens : tout s'echange.

    Les composantes selon `Oy` se detruisent et celles selon `Ox`
    s'ajoutent. Publie pour que la difference se **voie** : sans ce
    pendant, rien ne distinguerait une somme d'une difference, et les
    deux donneraient une expression non nulle.
    """
    distance = _distance_au_point(demi_ecart, ordonnee)
    module = Fil(courant=courant, distance=distance).module_du_champ()
    sinus = ordonnee / distance
    projection = module * demi_ecart / distance
    return -2 * module * sinus, projection - projection


def force_lineique_en_un_point(
    courant: sp.Expr,
    demi_ecart: sp.Expr,
    ordonnee: sp.Expr,
    courant_du_troisieme: sp.Expr,
    permeabilite: sp.Expr,
) -> tuple[sp.Expr, sp.Expr]:
    """`dF/dz = I_2 (-u_z) ^ mu_0 H`, le troisieme courant etant entrant.

    Le `-u_z` vient de ce que `I_2` entre dans le plan de la figure. Le
    produit vectoriel avec `u_z` echange les composantes et change un
    signe : `u_z ^ u_y = -u_x`. La force est donc orthogonale au champ,
    ce qu'un produit scalaire nul verifie.
    """
    champ_x, champ_y = champ_sur_la_mediatrice(courant, demi_ecart, ordonnee)
    facteur = permeabilite * courant_du_troisieme
    return facteur * champ_y, -facteur * champ_x


def produit_scalaire(
    premier: tuple[sp.Expr, sp.Expr],
    second: tuple[sp.Expr, sp.Expr],
) -> sp.Expr:
    """Le produit scalaire de deux vecteurs plans.

    Nomme plutot qu'ecrit en ligne dans le test : c'est le controle
    d'orthogonalite, et il a un sens physique propre.
    """
    return premier[0] * second[0] + premier[1] * second[1]
