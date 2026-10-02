"""Les distances et les sommes de l'exercice 8 -- premiere moitie du jeu.

Scinde de `macros_champ_08` quand celui-ci a franchi la centaine de
lignes. La coupure n'est pas arbitraire : d'un cote les **donnees** de
l'enonce et les deux sommes geometriques, de l'autre les **trois valeurs
du champ** et la convergence. Les premieres sont de la lecture, les
secondes du raisonnement.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.application.formatage import nombre
from polytech.electromagnetisme.domain.magnetostatique.nappe import (
    distances_geometriques,
    somme_des_inverses,
    somme_fermee,
)

COURANT = sp.Integer(3)
NOMBRE_DE_PAIRES = 10
DEPUIS_UN_CENTIMETRE = 0
DEPUIS_DEUX_CENTIMETRES = 1

DECIMALES_DE_LA_SOMME = 2
"""Deux pour la somme des inverses, dont la note de verification donne 99,9."""


def distances_de_la_figure() -> tuple[sp.Expr, ...]:
    """Les dix distances lues sur la figure : de 1 a 512 cm."""
    return distances_geometriques(DEPUIS_UN_CENTIMETRE, NOMBRE_DE_PAIRES)


def distances_de_la_somme() -> tuple[sp.Expr, ...]:
    """Les dix distances qu'ecrit la somme du manuscrit : de 2 a 1024 cm."""
    return distances_geometriques(DEPUIS_DEUX_CENTIMETRES, NOMBRE_DE_PAIRES)


def _en_centimetres(distance: sp.Expr) -> str:
    """Rend une distance en centimetres, sans decimale.

    Les distances de l'enonce sont des puissances de deux en
    centimetres : aucune n'a de partie fractionnaire une fois convertie.
    """
    return nombre(float(distance) * 100, 0)


def macros_de_l_enonce() -> dict[str, str]:
    """Le courant, le compte de paires, et les distances extremes."""
    figure = distances_de_la_figure()
    somme = distances_de_la_somme()
    return {
        "ChampHuitCourant": nombre(float(COURANT), 0),
        "ChampHuitPaires": str(NOMBRE_DE_PAIRES),
        "ChampHuitPremiereDistance": _en_centimetres(figure[0]),
        "ChampHuitDerniereDistance": _en_centimetres(figure[-1]),
        "ChampHuitDistanceDeLaSomme": _en_centimetres(somme[0]),
        "ChampHuitDerniereDeLaSomme": _en_centimetres(somme[-1]),
    }


def macros_des_sommes() -> dict[str, str]:
    """Les deux sommes des inverses, explicite et fermee."""
    return {
        "ChampHuitSommeDeLaFigure": nombre(
            float(somme_des_inverses(distances_de_la_figure())),
            DECIMALES_DE_LA_SOMME,
        ),
        "ChampHuitSommeDuManuscrit": nombre(
            float(somme_des_inverses(distances_de_la_somme())),
            DECIMALES_DE_LA_SOMME,
        ),
        "ChampHuitSommeFermee": nombre(
            float(somme_fermee(DEPUIS_DEUX_CENTIMETRES, NOMBRE_DE_PAIRES)),
            DECIMALES_DE_LA_SOMME,
        ),
    }
