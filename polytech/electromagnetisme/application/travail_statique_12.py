"""Statique 12 : le travail pour ecarter les armatures d'un condensateur plan.

L'enonce est l'exercice 4 de la section 3.1 du recueil
(`Electromagnetisme/EXS/Exs_Notes_manuscrites.pdf`, page imprimee 63) ;
le manuscrit qui le resout porte le numero 12.

Deux cas, et tout l'exercice est dans leur difference : a potentiels
constants la source reste branchee et la charge varie ; a charges
constantes le condensateur est isole et c'est la tension qui varie.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.electrostatique.images import PERMITTIVITE
from polytech.electromagnetisme.domain.energie_et_forces import (
    DEPLACEMENT,
    bilan_a_potentiels_constants,
    energie_a_charge_donnee,
    energie_d_un_condensateur,
    force_a_charges_constantes,
    force_a_potentiels_constants,
)

EPSILON: Final = PERMITTIVITE
SURFACE: Final = sp.Symbol("S", positive=True)
DEPART: Final = sp.Symbol("a", positive=True)
ARRIVEE: Final = sp.Symbol("b", positive=True)
TENSION: Final = sp.Symbol("U", positive=True)
CHARGE: Final = sp.Symbol("Q", positive=True)


@dataclass(frozen=True, slots=True)
class Ecartement:
    """L'energie avant et apres, la force en chemin, le travail a fournir."""

    initiale: sp.Expr
    finale: sp.Expr
    force: sp.Expr
    travail: sp.Expr


def capacite(distance: sp.Expr = DEPLACEMENT) -> sp.Expr:
    """`C = eps0 S / x`, les armatures etant dans le vide."""
    return EPSILON * SURFACE / distance


def charge_initiale() -> sp.Expr:
    """`Q = C_i U` : la charge prise a la source avant de s'en isoler."""
    return capacite(DEPART) * TENSION


def a_potentiels_constants() -> Ecartement:
    """La source reste branchee : `T = -(W_f - W_i)`."""
    energie = energie_d_un_condensateur(capacite(), TENSION)
    initiale, finale = (energie.subs(DEPLACEMENT, x) for x in (DEPART, ARRIVEE))
    force = force_a_potentiels_constants(energie)
    return Ecartement(initiale, finale, force, sp.simplify(initiale - finale))


def travail_en_fonction_de_la_charge() -> sp.Expr:
    """A charges constantes, avant de remplacer `Q` : `(b - a) Q^2 / (2 eps0 S)`."""
    energie = energie_a_charge_donnee(capacite(), CHARGE)
    return sp.simplify(
        energie.subs(DEPLACEMENT, ARRIVEE) - energie.subs(DEPLACEMENT, DEPART)
    )


def a_charges_constantes() -> Ecartement:
    """Le condensateur est isole : `T = +(W_f - W_i)`, avec `Q = C_i U`."""
    energie = energie_a_charge_donnee(capacite(), charge_initiale())
    initiale, finale = (energie.subs(DEPLACEMENT, x) for x in (DEPART, ARRIVEE))
    force = force_a_charges_constantes(energie)
    return Ecartement(initiale, finale, force, sp.simplify(finale - initiale))


def rapport_des_travaux() -> sp.Expr:
    """Ce qu'il en coute de plus quand le condensateur est isole."""
    isole, branche = a_charges_constantes().travail, a_potentiels_constants().travail
    return sp.simplify(isole / branche)


def bilan_de_la_source() -> tuple[sp.Expr, sp.Expr]:
    """A potentiels constants : ce que la source recoit, ce que le champ perd."""
    travail_du_champ = -a_potentiels_constants().travail
    gain_du_champ, variation_de_la_source = bilan_a_potentiels_constants(
        travail_du_champ
    )
    return sp.simplify(variation_de_la_source), sp.simplify(-gain_du_champ)


def tension_finale() -> sp.Expr:
    """A charges constantes : `U_f = Q / C_f`."""
    return sp.simplify(charge_initiale() / capacite(ARRIVEE))


def a_l_infini() -> tuple[sp.Expr, sp.Expr]:
    """Le travail pour ecarter indefiniment, dans chacun des deux cas."""
    branche = sp.limit(a_potentiels_constants().travail, ARRIVEE, sp.oo)
    return branche, sp.limit(a_charges_constantes().travail, ARRIVEE, sp.oo)
