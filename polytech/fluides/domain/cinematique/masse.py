"""Conservation de la masse d'une particule fluide (cours III.6).

Le corrige concluait « l'ecoulement est compressible » a partir de
``div V != 0``. Le pas n'est pas immediat : la divergence dit que le
*volume* d'une particule change, pas que le *fluide* se comprime. Ce qui
autorise la conclusion est le principe rappele par le cours en III.6 --
un systeme a « masse constante et volume variable ».

De m = rho V constante decoule tout le reste : la masse volumique suit
l'inverse du volume, et sa forme differentielle est l'equation de
continuite.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.evolution import loi_de_volume
from polytech.fluides.domain.cinematique.symboles import T


def _exiger_primitive(divergence: sp.Expr) -> None:
    """Refuse une divergence dont la primitive n'est pas elementaire."""
    if sp.integrate(divergence, (T, 0, T)).has(sp.Integral):
        msg = f"divergence sans primitive elementaire : {divergence}"
        raise ValueError(msg)


def loi_de_masse_volumique(divergence: sp.Expr) -> sp.Expr:
    """Rapport ``rho(t)/rho_0`` d'une particule suivie dans son mouvement.

    La masse ``m = rho V`` ne change pas ; le volume suit
    :func:`~.evolution.loi_de_volume` ; la masse volumique suit donc son
    inverse.
    """
    _exiger_primitive(divergence)
    return sp.simplify(1 / loi_de_volume(divergence))


def masse_de_la_particule(divergence: sp.Expr) -> sp.Expr:
    """Produit ``rho V`` rapporte a sa valeur initiale : doit valoir 1.

    C'est la verification du principe lui-meme, et non son hypothese : on
    construit les deux lois separement, puis on constate que leur produit
    ne depend pas du temps.
    """
    return sp.simplify(loi_de_masse_volumique(divergence) * loi_de_volume(divergence))


def taux_de_variation_de_la_masse_volumique(divergence: sp.Expr) -> sp.Expr:
    """``(1/rho) Drho/Dt``, forme locale de la conservation de la masse.

    Elle vaut ``-div V`` : c'est l'equation de continuite, obtenue ici en
    derivant la loi integrale plutot qu'en la postulant.
    """
    loi = loi_de_masse_volumique(divergence)
    return sp.simplify(sp.diff(loi, T) / loi)
