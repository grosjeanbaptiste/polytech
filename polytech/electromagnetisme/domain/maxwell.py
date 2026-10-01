"""Les quatre equations de Maxwell, et ce qu'elles impliquent toutes seules.

Le chapitre IV.1 les pose comme la synthese des trois chapitres
precedents : chacune generalise une loi deja vue, en lui ajoutant le terme
que le regime variable fait apparaitre.

Ce que la bibliotheque porte n'est pas la copie des quatre lignes -- une
copie ne se verifie pas -- mais ce qui s'en **deduit** : l'equation de
continuite `div J + d eta/dt = 0` sort de la loi d'Ampere-Maxwell et de
celle de Gauss sans hypothese supplementaire. C'est la verification la
plus forte disponible sur ce chapitre, parce qu'elle echoue si l'un des
deux termes est mal ecrit.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.lois import (
    divergence_cartesienne,
    rotationnel_cartesien,
)
from polytech.electromagnetisme.domain.symboles import TEMPS

Champ = tuple[sp.Expr, sp.Expr, sp.Expr]


def residu_de_gauss_electrique(deplacement: Champ, densite: sp.Expr) -> sp.Expr:
    """`div D - eta`, qui doit valoir zero.

    Les quatre lois sont exposees comme des **residus** et non comme des
    `sp.Eq`. La raison est concrete : `sp.Eq(0, 1)` ne rend pas une
    equation mais un booleen, et un appelant qui lirait son `rhs`
    tomberait sur un `AttributeError` dans le cas meme ou la loi est
    violee -- c'est-a-dire exactement quand on a besoin de savoir.
    """
    return sp.simplify(divergence_cartesienne(deplacement) - densite)


def residu_de_gauss_magnetique(induction: Champ) -> sp.Expr:
    """`div B`, qui doit valoir zero : pas de charge magnetique."""
    return divergence_cartesienne(induction)


def residus_de_faraday(champ_electrique: Champ, induction: Champ) -> Champ:
    """`rot E + dB/dt`, composante par composante, chacune a annuler."""
    rotationnel = rotationnel_cartesien(champ_electrique)
    return tuple(
        sp.simplify(composante + sp.diff(derivee, TEMPS))
        for composante, derivee in zip(rotationnel, induction, strict=True)
    )


def residus_d_ampere_maxwell(
    champ_magnetique: Champ,
    densite_de_courant: Champ,
    deplacement: Champ,
) -> Champ:
    """`rot H - J - dD/dt` : le dernier terme est l'apport de Maxwell."""
    rotationnel = rotationnel_cartesien(champ_magnetique)
    return tuple(
        sp.simplify(composante - courant - sp.diff(derivee, TEMPS))
        for composante, courant, derivee in zip(
            rotationnel,
            densite_de_courant,
            deplacement,
            strict=True,
        )
    )


def equation_de_continuite(
    densite_de_courant: Champ,
    densite_de_charge: sp.Expr,
) -> sp.Expr:
    """`div J + d eta/dt`, qui doit valoir zero.

    Elle n'est pas un postulat : elle se deduit d'Ampere-Maxwell en
    prenant la divergence des deux membres -- la divergence d'un
    rotationnel est nulle, donc `div J + d(div D)/dt = 0`, et `div D` vaut
    `eta` par Gauss. L'exposer comme une expression a annuler permet de la
    **verifier** sur un couple donne.
    """
    return sp.simplify(
        divergence_cartesienne(densite_de_courant) + sp.diff(densite_de_charge, TEMPS),
    )


def conserve_la_charge(
    densite_de_courant: Champ,
    densite_de_charge: sp.Expr,
) -> bool:
    """Vrai si le couple (J, eta) satisfait l'equation de continuite."""
    return bool(equation_de_continuite(densite_de_courant, densite_de_charge) == 0)
