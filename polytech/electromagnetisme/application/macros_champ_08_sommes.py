"""Les largeurs, les distances et les sommes de l'exercice 8.

Scinde de `macros_champ_08` quand celui-ci a franchi la centaine de
lignes. La coupure n'est pas arbitraire : d'un cote les **donnees** de
l'enonce et les sommes geometriques, de l'autre les **valeurs du champ**
et la convergence. Les premieres sont de la lecture, les secondes du
raisonnement.

L'enonce donne des **largeurs** de spires ; les distances a l'axe en
sont les moities. Les deux suites sont publiees, parce que le seul piege
de l'exercice est de prendre l'une pour l'autre.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.domain.magnetostatique.nappe import (
    demi_largeurs,
    largeurs_des_spires,
    somme_des_inverses,
    somme_fermee,
)

COURANT = sp.Integer(3)
NOMBRE_DE_SPIRES = 10
DEPUIS_UN_CENTIMETRE = 0
"""L'exposant de la premiere distance a l'axe : un centimetre."""

DECIMALES_DE_LA_SOMME = 2


def largeurs_de_l_enonce() -> tuple[sp.Expr, ...]:
    """Les dix largeurs que donne le recueil : de 2 a 1024 cm."""
    return largeurs_des_spires(NOMBRE_DE_SPIRES)


def distances_a_l_axe() -> tuple[sp.Expr, ...]:
    """Les dix demi-largeurs : de 1 a 512 cm."""
    return demi_largeurs(largeurs_de_l_enonce())


def _en_centimetres(longueur: sp.Expr) -> str:
    """Rend une longueur en centimetres, sans decimale.

    Largeurs et distances sont des puissances de deux en centimetres :
    aucune n'a de partie fractionnaire une fois convertie.
    """
    return nombre(float(longueur) * 100, 0)


def macros_de_l_enonce() -> dict[str, str]:
    """Le courant, le compte de spires, les largeurs et distances extremes."""
    largeurs = largeurs_de_l_enonce()
    distances = distances_a_l_axe()
    return {
        "ChampHuitCourant": nombre(float(COURANT), 0),
        "ChampHuitSpires": str(NOMBRE_DE_SPIRES),
        "ChampHuitPremiereLargeur": _en_centimetres(largeurs[0]),
        "ChampHuitDerniereLargeur": _en_centimetres(largeurs[-1]),
        "ChampHuitPremiereDistance": _en_centimetres(distances[0]),
        "ChampHuitDerniereDistance": _en_centimetres(distances[-1]),
    }


def macros_des_sommes() -> dict[str, str]:
    """Les sommes des inverses : des distances, des largeurs, et la fermee."""
    return {
        "ChampHuitSommeDesDistances": nombre(
            float(somme_des_inverses(distances_a_l_axe())),
            DECIMALES_DE_LA_SOMME,
        ),
        "ChampHuitSommeDesLargeurs": nombre(
            float(somme_des_inverses(largeurs_de_l_enonce())),
            DECIMALES_DE_LA_SOMME,
        ),
        "ChampHuitSommeFermee": nombre(
            float(somme_fermee(DEPUIS_UN_CENTIMETRE, NOMBRE_DE_SPIRES)),
            DECIMALES_DE_LA_SOMME,
        ),
    }
