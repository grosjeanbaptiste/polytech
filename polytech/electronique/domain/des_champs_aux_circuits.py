"""Ce que le cours emprunte a l'electromagnetisme pour fonder les circuits.

Le chapitre 1 ne refait pas un cours d'electromagnetisme : il justifie la
theorie des circuits. Et il la justifie par **un seul critere**, la
longueur electrique.

`theta = 2 pi l / lambda` avec `lambda = v / f` est le dephasage que le
delai de propagation induit. La planche donne la regle en clair : les
delais deviennent negligeables quand `l_c < lambda / 20`, ou `l_c` est le
plus long chemin du circuit. En dessous, les lois de Kirchhoff valent ; au
dela, il faut une ligne de transmission -- et tout le reste du cours
devient faux.

Deux autres apports du chapitre sont calculables : la conductivite de
Drude, qui dit **d'ou vient** une resistance, et l'estimation d'une
capacite parasite, qui dit pourquoi un circuit reel a des poles que son
schema n'a pas.

Ce module ne duplique pas `polytech.electromagnetisme` : il porte l'usage
que l'electronique fait de ces lois, et cet usage-la n'existe pas dans
l'autre contexte. Les deux restent etanches, comme la regle des contextes
bornes l'exige.
"""

from __future__ import annotations

import math
from typing import Final

FRACTION_DE_LONGUEUR_D_ONDE: Final = 20
"""La regle de la planche : `l_c < lambda / 20`.

Vingt et non dix : c'est la valeur que le cours ecrit, et la changer
changerait la frequence a laquelle il faut abandonner Kirchhoff.
"""

CHARGE_DE_L_ELECTRON: Final = 1.602176634e-19
"""`q_e`, en coulombs."""

MASSE_DE_L_ELECTRON: Final = 9.1093837015e-31
"""`m_e` au repos, en kilogrammes."""


def longueur_d_onde(vitesse: float, frequence: float) -> float:
    """`lambda = v / f`, en metres."""
    if frequence <= 0:
        msg = f"frequence nulle ou negative : {frequence} Hz"
        raise ValueError(msg)
    if vitesse <= 0:
        msg = f"vitesse de propagation nulle ou negative : {vitesse} m/s"
        raise ValueError(msg)
    return vitesse / frequence


def longueur_electrique(longueur: float, vitesse: float, frequence: float) -> float:
    """`2 pi l / lambda`, en radians : le dephasage du au delai.

    C'est un **angle**, et non une longueur, malgre son nom. La planche
    insiste sur l'unite parce que l'oubli est courant.
    """
    if longueur < 0:
        msg = f"longueur de chemin negative : {longueur} m"
        raise ValueError(msg)
    return 2 * math.pi * longueur / longueur_d_onde(vitesse, frequence)


def les_constantes_localisees_valent(
    longueur: float,
    vitesse: float,
    frequence: float,
) -> bool:
    """Vrai si `l < lambda / 20` : le modele a constantes localisees tient.

    C'est le critere qui autorise tout le reste du cours. En dessous, les
    lois de Kirchhoff valent ; au-dela, il faut une ligne de transmission,
    et un corrige qui appliquerait quand meme l'analyse nodale serait faux
    sans le paraitre.
    """
    if longueur < 0:
        msg = f"longueur de chemin negative : {longueur} m"
        raise ValueError(msg)
    return longueur < longueur_d_onde(vitesse, frequence) / FRACTION_DE_LONGUEUR_D_ONDE


def frequence_limite(longueur: float, vitesse: float) -> float:
    """La frequence au-dela de laquelle le critere ci-dessus cesse de tenir.

    `f_max = v / (20 l)` : l'inverse du critere, et c'est sous cette forme
    qu'un dimensionnement l'emploie -- on connait la taille du circuit, on
    cherche jusqu'ou on peut monter.
    """
    if longueur <= 0:
        msg = f"longueur de chemin nulle ou negative : {longueur} m"
        raise ValueError(msg)
    if vitesse <= 0:
        msg = f"vitesse de propagation nulle ou negative : {vitesse} m/s"
        raise ValueError(msg)
    return vitesse / (FRACTION_DE_LONGUEUR_D_ONDE * longueur)


def conductivite_de_drude(
    densite_electronique: float,
    temps_de_vol_libre: float,
) -> float:
    """`sigma = n q_e^2 tau / m_e`, en S/m.

    La planche le dit elle-meme : le modele est « standard, introductif,
    mais depasse » -- il rend des libres parcours moyens faux. Il est ici
    parce que le cours s'en sert pour dire **d'ou vient** une resistance,
    pas pour en calculer une.
    """
    if densite_electronique <= 0 or temps_de_vol_libre <= 0:
        msg = (
            f"densite electronique ou temps de vol libre non positif : "
            f"{densite_electronique}, {temps_de_vol_libre}"
        )
        raise ValueError(msg)
    numerateur = densite_electronique * CHARGE_DE_L_ELECTRON**2 * temps_de_vol_libre
    return numerateur / MASSE_DE_L_ELECTRON


def capacite_parasite(permittivite: float, aire: float, distance: float) -> float:
    """`epsilon S / e` : l'estimation que la planche fait d'une capacite cachee.

    Elle explique pourquoi un circuit reel a des poles que son schema n'a
    pas -- et pourquoi ces poles montent quand on rapproche les pistes.
    """
    if aire <= 0 or distance <= 0:
        msg = f"geometrie invalide : aire {aire} m^2, distance {distance} m"
        raise ValueError(msg)
    if permittivite <= 0:
        msg = f"permittivite nulle ou negative : {permittivite} F/m"
        raise ValueError(msg)
    return permittivite * aire / distance


def kirchhoff_des_courants(courants: tuple[float, ...]) -> float:
    """`somme I` a un noeud, qui doit valoir zero.

    Le fascicule en donne la **preuve** a partir de l'equation de
    continuite : la charge ne s'accumule pas en un point, donc ce qui entre
    sort. La loi n'est pas un postulat, elle est une consequence.
    """
    if not courants:
        msg = "aucune branche au noeud : la loi des courants n'a pas d'objet"
        raise ValueError(msg)
    return sum(courants)


def kirchhoff_des_tensions(tensions: tuple[float, ...]) -> float:
    """`somme U` sur une maille, qui doit valoir zero.

    Sa preuve est l'existence du potentiel : le champ electrostatique est
    a circulation nulle, donc la somme des differences de potentiel sur un
    contour ferme s'annule. C'est pourquoi la loi tombe des que la
    longueur electrique cesse d'etre negligeable -- un champ induit n'est
    plus a circulation nulle.
    """
    if not tensions:
        msg = "aucune branche dans la maille : la loi des tensions n'a pas d'objet"
        raise ValueError(msg)
    return sum(tensions)


def inductance_cachee_d_une_boucle(aire: float, perimetre: float) -> float:
    """Une estimation de l'inductance qu'aucun schema ne montre.

    `mu_0 (aire / perimetre) x 2 pi` en ordre de grandeur. Le fascicule
    l'appelle « hidden inductance » et c'est elle qui fixe la vitesse a
    laquelle un courant peut changer -- d'ou la regle de reduire les
    boucles de courant sur un circuit imprime.
    """
    if aire <= 0:
        msg = f"aire nulle ou negative : {aire} m^2"
        raise ValueError(msg)
    if perimetre <= 0:
        msg = f"perimetre nul ou negatif : {perimetre} m"
        raise ValueError(msg)
    permeabilite = 4 * math.pi * 1e-7
    return permeabilite * 2 * math.pi * aire / perimetre


def tension_de_defaut_terre_neutre(courant: float, resistance: float) -> float:
    """`R I` : la tension qui apparait entre la terre et le neutre.

    « Earth and neutral wire » : ils ne sont pas au meme potentiel. Le
    courant de retour traverse la resistance du neutre et y cree une
    chute -- raison pour laquelle une masse reliee au neutre **n'est pas**
    une masse, et pour laquelle une mesure referencee au neutre porte
    cette tension en mode commun.
    """
    if courant < 0:
        msg = f"courant negatif : {courant} A -- employer son module"
        raise ValueError(msg)
    if resistance < 0:
        msg = f"resistance negative : {resistance} ohm"
        raise ValueError(msg)
    return courant * resistance
