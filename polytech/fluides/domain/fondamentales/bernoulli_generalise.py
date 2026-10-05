"""L'equation de Bernoulli generalisee : pertes, profils, pompes et turbines.

Chapitre IV.5 des planches. Le long d'une ligne de courant d'un ecoulement
stationnaire, incompressible et sans frottement, la pression totale
`rho V^2 / 2 + p + rho g z` se conserve : pression dynamique, pression
statique et pression hydrostatique. Au point d'arret, toute la pression
dynamique s'ajoute a la pression statique.

Trois generalisations :

- les fluides visqueux perdent de la charge par frottement, un terme
  toujours positif ;
- un profil de vitesse non uniforme porte plus d'energie cinetique que sa
  vitesse moyenne ne le dit, d'un facteur `alpha = <v^3> / <v>^3` : un pour
  un profil plat, deux pour un profil parabolique ;
- une pompe apporte du travail, une turbine en preleve, chacune avec son
  rendement.

Les bilans sont ecrits en hauteurs, c'est-a-dire par unite de poids.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.grandeurs import PESANTEUR


def pression_d_arret(statique: float, masse_volumique: float, vitesse: float) -> float:
    """`p_A = p_inf + rho v^2 / 2`, en pascals."""
    if masse_volumique <= 0:
        msg = f"masse volumique nulle ou negative : {masse_volumique} kg/m^3"
        raise ValueError(msg)
    return statique + masse_volumique * vitesse**2 / 2


def facteur_de_correction_cinetique(
    profil: sp.Expr, rayon: sp.Symbol, paroi: sp.Expr
) -> sp.Expr:
    """`alpha = <v^3> / <v>^3`, les moyennes etant prises sur un disque."""

    def moyenne(grandeur: sp.Expr) -> sp.Expr:
        return 2 * sp.integrate(grandeur * rayon, (rayon, 0, paroi)) / paroi**2

    debitante = sp.simplify(moyenne(sp.sympify(profil)))
    if debitante == 0:
        msg = "vitesse moyenne nulle : le facteur de correction n'est pas defini"
        raise ValueError(msg)
    return sp.simplify(moyenne(sp.sympify(profil) ** 3) / debitante**3)


def hauteur_manometrique(
    *,
    denivele: float,
    vitesse_finale: float,
    vitesse_initiale: float = 0.0,
    hauteur_de_pression: float = 0.0,
    pertes: float = 0.0,
) -> float:
    """`(z_2 - z_1) + (V_2^2 - V_1^2) / (2 g) + (p_2 - p_1) / (rho g) + h_L`, en m.

    L'ecart de pression est donne en hauteur de fluide, `(p_2 - p_1) / (rho g)`.
    """
    cinetique = (vitesse_finale**2 - vitesse_initiale**2) / (2 * PESANTEUR)
    return denivele + cinetique + hauteur_de_pression + pertes


def puissance_hydraulique(
    masse_volumique: float, debit: float, hauteur: float
) -> float:
    """`rho g Q H`, en watts : ce que le fluide echange sur cette hauteur."""
    return masse_volumique * PESANTEUR * debit * hauteur


def rendement(utile: float, fournie: float) -> float:
    """Le quotient de la puissance utile par la puissance fournie."""
    if fournie <= 0:
        msg = f"puissance fournie nulle ou negative : {fournie} W"
        raise ValueError(msg)
    if utile > fournie:
        msg = f"plus que fourni : {utile} W tires de {fournie} W"
        raise ValueError(msg)
    return utile / fournie


def puissance_d_une_turbine(
    masse_volumique: float,
    *,
    debit: float,
    chute: float,
    pertes: float,
    rendement_global: float,
) -> float:
    """`eta rho g Q (H - h_L)`, en watts."""
    if pertes > chute:
        msg = f"pertes superieures a la chute : {pertes} m pour {chute} m"
        raise ValueError(msg)
    if not 0 < rendement_global <= 1:
        msg = f"rendement hors de ]0 ; 1] : {rendement_global}"
        raise ValueError(msg)
    nette = puissance_hydraulique(masse_volumique, debit, chute - pertes)
    return rendement_global * nette


def puissance_a_l_arbre_d_une_pompe(
    masse_volumique: float, debit: float, hauteur: float, *, rendement_pompe: float
) -> float:
    """`rho g Q H / eta`, en watts."""
    if not 0 < rendement_pompe <= 1:
        msg = f"rendement hors de ]0 ; 1] : {rendement_pompe}"
        raise ValueError(msg)
    return puissance_hydraulique(masse_volumique, debit, hauteur) / rendement_pompe
