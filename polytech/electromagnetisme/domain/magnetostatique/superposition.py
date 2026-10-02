"""Superposition de trois fils, et la force de Laplace qui en resulte.

Exercice 11 du chapitre IV, entierement symbolique : l'enonce ne donne
aucun nombre. C'est ce qui en fait le plus severe du chapitre -- il n'y a
pas d'application numerique pour rattraper une erreur de signe.

Trois choses s'y jouent.

**La superposition porte sur des vecteurs.** Les trois fils contribuent
chacun `I / (2 pi d)` en module, mais leurs **directions** differrent :
deux portent `u_y` en sens opposes, le troisieme `-u_x`. La composante
`u_y` est donc une **difference**, et c'est ce qui permet au champ de
s'annuler -- `I_1/d_1 = I_3/d_3`. Additionner les modules donnerait un
champ qui ne s'annule jamais, ce qui est faux et physiquement absurde :
c'est ainsi qu'on neutralise un champ parasite dans un cablage.

**Le produit vectoriel tourne la force.** `u_z ^ u_x = u_y` et
`u_z ^ u_y = -u_x` : les composantes s'echangent en changeant un signe,
et la force est orthogonale au champ. C'est la que les fautes passent le
plus facilement, et le test la verifie par un produit scalaire nul --
controle independant des expressions elles-memes.

**Le terme croise du module est negatif.** En developpant le carre de la
difference apparait `-2 I_1 I_3 / (d_1 d_3)`, et ce signe porte toute
l'information : les fils 1 et 3 s'opposent. Avec un `+`, l'expression
resterait toujours superieure a la contribution du seul fil 2, donc ne
pourrait jamais l'atteindre -- alors que le vrai module l'atteint
exactement quand les deux autres se compensent.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True, slots=True)
class Fil:
    """Un conducteur rectiligne infini, vu du point `P`.

    Deux grandeurs et pas trois arguments : un fil **est** un courant a
    une distance, et les separer dans une signature laissait croire a six
    donnees independantes la ou il y a trois objets.
    """

    courant: sp.Expr
    distance: sp.Expr

    def module_du_champ(self) -> sp.Expr:
        """`I / (2 pi d)` : le module que Biot et Savart donnent."""
        return self.courant / (2 * sp.pi * self.distance)


def champ_de_trois_fils(
    premier: Fil,
    deuxieme: Fil,
    troisieme: Fil,
) -> tuple[sp.Expr, sp.Expr]:
    """Le champ resultant au point `P`, en composantes `(H_x, H_y)`.

    Les directions sont celles de la figure de l'enonce : le premier fil
    porte `+u_y`, le deuxieme `-u_x`, le troisieme `-u_y`. La composante
    `u_y` est donc la **difference** des contributions 1 et 3.
    """
    champ_x = -deuxieme.module_du_champ()
    champ_y = premier.module_du_champ() - troisieme.module_du_champ()
    return champ_x, champ_y


def force_par_unite_de_longueur(
    courant: sp.Expr,
    champ: tuple[sp.Expr, sp.Expr],
    permeabilite: sp.Expr,
) -> tuple[sp.Expr, sp.Expr]:
    """`dF/dz = mu_0 I (u_z ^ H)`, en composantes `(F_x, F_y)`.

    Le produit vectoriel avec `u_z` echange les composantes et change un
    signe : `u_z ^ u_x = u_y`, `u_z ^ u_y = -u_x`. La force est donc
    orthogonale au champ, comme toute force de Laplace sur un conducteur
    porte par `u_z`.
    """
    champ_x, champ_y = champ
    return -permeabilite * courant * champ_y, permeabilite * courant * champ_x


def module_de_la_force(
    courant: sp.Expr,
    champ: tuple[sp.Expr, sp.Expr],
    permeabilite: sp.Expr,
) -> sp.Expr:
    """Le module de la force par unite de longueur.

    Conserve la forme compacte -- la racine de la somme des carres --
    plutot que de developper : le developpement est l'affaire du
    document, et c'est lui qui fait apparaitre le terme croise negatif.
    """
    force_x, force_y = force_par_unite_de_longueur(courant, champ, permeabilite)
    return sp.sqrt(force_x**2 + force_y**2)
