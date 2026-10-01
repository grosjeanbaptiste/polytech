"""Le facteur de bruit, la figure de bruit, et ce qu'ils servent a decider.

Separe de `bruit`, qui porte les **sources**, parce que ce module porte la
**metrologie** : comment on mesure un bruit, comment on le compare, et
comment on choisit un composant. Les deux responsabilites sont distinctes,
et le fichier unique depassait la limite de taille du depot.

Le fil directeur est une inegalite : « noise factor should always be
>= 1 », ecrit la planche. Un quadripole n'ameliore jamais un rapport
signal sur bruit ; au mieux il le preserve. Tout le reste -- la figure en
decibels, la temperature de bruit equivalente, la plage dynamique, le
choix d'une technologie -- se lit a partir de la.
"""

from __future__ import annotations

import math
from typing import Final

DECIBELS_PAR_DECADE: Final = 10
"""La figure de bruit est un rapport de **puissances** : 10 log, et non 20."""

DECIBELS_DE_TENSION: Final = 20
"""La plage dynamique compare des tensions : 20 log."""

TEMPERATURE_DE_REFERENCE: Final = 290.0
"""`T_0`, en kelvins. Convention de l'IEEE pour la temperature de bruit."""

FACTEUR_MINIMAL_DE_BRUIT: Final = 1.0
"""Un quadripole degrade le rapport signal sur bruit, ou le preserve."""

TECHNOLOGIES: Final = ("bipolaire", "JFET", "CMOS")
"""Les trois technologies que le fascicule compare.

Le bipolaire a le plus faible bruit en tension, le JFET le plus faible
bruit en courant, le CMOS est entre les deux et moins cher. Le choix se
fait donc sur l'impedance de la **source**, et non sur une qualite absolue.
"""


def facteur_de_bruit(rapport_a_l_entree: float, rapport_a_la_sortie: float) -> float:
    """`F = SNR_entree / SNR_sortie`, qui ne descend pas sous l'unite."""
    if rapport_a_la_sortie <= 0 or rapport_a_l_entree <= 0:
        msg = (
            f"rapport signal sur bruit nul ou negatif : {rapport_a_l_entree} "
            f"et {rapport_a_la_sortie}"
        )
        raise ValueError(msg)
    facteur = rapport_a_l_entree / rapport_a_la_sortie
    if facteur < FACTEUR_MINIMAL_DE_BRUIT:
        msg = (
            f"facteur de bruit sous l'unite : {facteur:.4g} -- un quadripole "
            f"n'ameliore pas un rapport signal sur bruit"
        )
        raise ValueError(msg)
    return facteur


def figure_de_bruit(facteur: float) -> float:
    """`NF = 10 log10 F`, en decibels. Dix et non vingt : des puissances."""
    if facteur < FACTEUR_MINIMAL_DE_BRUIT:
        msg = f"facteur de bruit sous l'unite : {facteur}"
        raise ValueError(msg)
    return DECIBELS_PAR_DECADE * math.log10(facteur)


def temperature_de_bruit_equivalente(
    facteur: float,
    reference: float = TEMPERATURE_DE_REFERENCE,
) -> float:
    """`T_e = (F - 1) T_0` : le bruit du quadripole vu comme une temperature."""
    if facteur < FACTEUR_MINIMAL_DE_BRUIT:
        msg = f"facteur de bruit sous l'unite : {facteur}"
        raise ValueError(msg)
    return (facteur - 1) * reference


def facteur_de_bruit_ponctuel(
    densite_totale: float,
    densite_de_source: float,
) -> float:
    """Le facteur de bruit **a une frequence donnee** -- le « spot noise factor ».

    Il depend de la frequence, et c'est pourquoi une fiche technique en
    donne une courbe et non un nombre : le bruit en 1/f le fait exploser
    aux basses frequences.
    """
    if densite_de_source <= 0:
        msg = f"densite de la source nulle ou negative : {densite_de_source}"
        raise ValueError(msg)
    return densite_totale / densite_de_source


def tension_de_bruit_depuis_une_fiche_technique(
    densite_en_nanovolts: float,
    bande: float,
) -> float:
    """De `nV/racine(Hz)` a la tension efficace, en volts.

    C'est le geste que tout dimensionnement de chaine commence, et l'unite
    en racine de hertz est ce qui le rend contre-intuitif.
    """
    if bande <= 0:
        msg = f"bande nulle ou negative : {bande} Hz"
        raise ValueError(msg)
    if densite_en_nanovolts < 0:
        msg = f"densite negative : {densite_en_nanovolts} nV/racine(Hz)"
        raise ValueError(msg)
    return densite_en_nanovolts * 1e-9 * math.sqrt(bande)


def bruit_dominant_selon_la_source(
    densite_en_tension: float,
    densite_en_courant: float,
    resistance_de_source: float,
) -> str:
    """Lequel des deux bruits domine : `tension` ou `courant`.

    C'est le critere de choix d'une technologie. Sur une source de forte
    impedance le bruit en courant domine -- il faut alors un JFET -- et sur
    une source de faible impedance c'est le bruit en tension, et un
    bipolaire convient mieux.
    """
    if resistance_de_source < 0:
        msg = f"resistance de source negative : {resistance_de_source} ohm"
        raise ValueError(msg)
    du_courant = densite_en_courant * resistance_de_source
    return "courant" if du_courant > densite_en_tension else "tension"


def plage_dynamique_en_decibels(
    pleine_echelle: float,
    tension_de_bruit: float,
) -> float:
    """`20 log10(V_max / V_bruit)` : ce que tout le chapitre sert a calculer.

    Bornee en haut par l'ecretage et en bas par le bruit, elle dit combien
    de bits sont **utiles** dans une chaine d'acquisition. Au-dela, les
    bits de poids faible ne codent que du bruit.
    """
    if tension_de_bruit <= 0:
        msg = f"tension de bruit nulle ou negative : {tension_de_bruit} V"
        raise ValueError(msg)
    if pleine_echelle <= 0:
        msg = f"pleine echelle nulle ou negative : {pleine_echelle} V"
        raise ValueError(msg)
    return DECIBELS_DE_TENSION * math.log10(pleine_echelle / tension_de_bruit)
