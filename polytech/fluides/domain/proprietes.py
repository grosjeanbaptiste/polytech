"""Les proprietes d'un fluide : viscosite, compressibilite, dilatation, tension.

Chapitre I du cours, et il n'etait **pas declare du tout** -- ni couvert,
ni a faire, ni silence motive. Exactement la situation de `TH-01` en
thermique : un fascicule intitule « Introduction » qu'on avait pris pour
une presentation sans l'ouvrir.

Les deux pieges du chapitre, et les deux sont des confusions d'unites.

**Les deux viscosites.** La dynamique `mu` est en `kg/(m s)`, la
cinematique `nu = mu / rho` en `m^2/s`. La planche les met en regard sur
l'air a vingt degres : a une atmosphere `mu = 1,83 . 10^-5` et
`nu = 1,52 . 10^-5` ; a quatre atmospheres `mu` ne change **pas** et
`nu` tombe a `0,380 . 10^-5`. La viscosite dynamique d'un gaz ne depend
donc pas de la pression ; la cinematique en depend par la masse
volumique, et le rapport `1,52/0,380` vaut exactement quatre.

**Les deux compressibilites.** Le coefficient `kappa` est en pascals --
ce qui surprend d'une « compressibilite » -- et vaut `P` pour un gaz
parfait. Son inverse `alpha`, la compressibilite isotherme, est en
pascals a la puissance moins un. Un gaz fortement compressible a donc un
`kappa` petit et un `alpha` grand.

La tension superficielle, elle, est deux grandeurs d'un seul nombre :
une force par unite de longueur et une energie par unite d'aire. La
planche l'ecrit `N.m/m^2 -> J/m^2`, et c'est ce qui autorise a la lire
comme « le travail d'expansion qu'il faut faire pour augmenter l'aire de
la surface liquide d'une unite ».
"""

from __future__ import annotations


def viscosite_cinematique(dynamique: float, masse_volumique: float) -> float:
    """`nu = mu / rho`, en m^2/s.

    C'est elle qui entre dans le nombre de Reynolds, et c'est la raison
    pour laquelle un ecoulement de gaz comprime est plus turbulent a
    vitesse egale : `nu` y est plus petit.
    """
    if masse_volumique <= 0:
        msg = f"masse volumique nulle ou negative : {masse_volumique} kg/m^3"
        raise ValueError(msg)
    return dynamique / masse_volumique


def coefficient_de_compressibilite_du_gaz_parfait(pression: float) -> float:
    """`kappa_gp = P`, en pascals.

    Le resultat tombe de `P = rho R T` : a temperature constante
    `(dP/drho)_T = R T = P / rho`, donc `kappa = rho (dP/drho)_T = P`. La
    planche l'ecrit tel quel.
    """
    if pression <= 0:
        msg = f"pression nulle ou negative : {pression} Pa"
        raise ValueError(msg)
    return pression


def compressibilite_isotherme(coefficient: float) -> float:
    """`alpha = 1 / kappa`, en Pa^-1 : la meme chose en sens inverse."""
    if coefficient <= 0:
        msg = f"pression nulle ou negative : {coefficient} Pa"
        raise ValueError(msg)
    return 1.0 / coefficient


def coefficient_de_dilatation_du_gaz_parfait(temperature: float) -> float:
    """`beta_gp = 1 / T`, en K^-1.

    A temperature ambiante il vaut environ `1/293`, soit 0,34 % par
    kelvin. C'est l'ordre de grandeur qui gouverne la convection
    naturelle, ou `beta` apparait dans le nombre de Grashof.
    """
    if temperature <= 0:
        msg = f"temperature absolue nulle ou negative : {temperature} K"
        raise ValueError(msg)
    return 1.0 / temperature


def energie_de_surface(tension: float, aire: float) -> float:
    """`sigma_s . A`, en joules : le travail d'expansion de la surface.

    Numeriquement, une tension en `N/m` est une energie surfacique en
    `J/m^2` : l'egalite est dimensionnelle, et c'est elle qui fait de la
    tension superficielle deux grandeurs d'un seul nombre.
    """
    if tension < 0:
        msg = f"tension superficielle negative : {tension} N/m"
        raise ValueError(msg)
    if aire < 0:
        msg = f"aire negative : {aire} m^2"
        raise ValueError(msg)
    return tension * aire
