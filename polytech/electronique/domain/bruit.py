"""Le bruit thermique d'une resistance (bruit de Johnson-Nyquist).

La densite spectrale de puissance se donne sous deux conventions. La
version *unilaterale*, celle du cours, ne compte que les frequences
positives et vaut donc le double de la bilaterale : c'est la distinction
que le mot « one-sided » de l'enonce recouvre.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sympy as sp

FACTEUR_UNILATERAL = 4
"""Deux pour la convention unilaterale, deux pour le theoreme de Nyquist."""


def densite_spectrale_unilaterale(
    resistance: sp.Expr,
    temperature: sp.Expr,
    constante_de_boltzmann: sp.Expr,
) -> sp.Expr:
    """Rend `4 k_B R T`, en volts carres par hertz."""
    return FACTEUR_UNILATERAL * constante_de_boltzmann * resistance * temperature


CHARGE_ELEMENTAIRE = 1.602176634e-19
"""`q`, en coulombs. Le bruit de grenaille la fait apparaitre explicitement."""

FACTEUR_DE_GRENAILLE = 2
"""`2 q I` : un pour la convention unilaterale, un pour le theoreme de Schottky."""

DECIBELS_PAR_DECADE = 10
"""La figure de bruit est un rapport de **puissances** : 10 log, et non 20."""

TEMPERATURE_DE_REFERENCE = 290.0
"""`T_0`, en kelvins. Convention de l'IEEE pour la temperature de bruit."""

FACTEUR_MINIMAL_DE_BRUIT = 1.0
"""Un quadripole degrade le rapport signal sur bruit, ou le preserve."""


def densite_spectrale_de_grenaille(
    courant: float,
    charge: float = CHARGE_ELEMENTAIRE,
) -> float:
    """`2 q I`, en amperes carres par hertz.

    Le bruit de grenaille n'apparait que lorsqu'un courant franchit une
    **barriere de potentiel** -- la planche y insiste. Un courant dans une
    resistance ohmique n'en produit pas : il ne franchit rien.
    """
    if courant < 0:
        msg = f"courant negatif : {courant} A -- le sens ne change pas le bruit"
        raise ValueError(msg)
    return FACTEUR_DE_GRENAILLE * charge * courant


def densite_spectrale_en_un_sur_f(
    coefficient: float,
    frequence: float,
    exposant: float = 1.0,
) -> float:
    """`K / f^alpha` : le bruit de scintillement, dit aussi rose ou d'excès.

    Il diverge en zero, et ce n'est pas un defaut du modele : c'est la
    raison pour laquelle on ne l'integre jamais depuis zero mais depuis une
    frequence basse que la duree de mesure fixe.
    """
    if frequence <= 0:
        msg = (
            f"frequence nulle ou negative : {frequence} Hz -- la densite en "
            f"1/f y diverge, et c'est pourquoi on integre depuis f_basse"
        )
        raise ValueError(msg)
    return coefficient / float(frequence**exposant)


def puissance_integree_blanc_et_rose(
    densite_blanche: float,
    coefficient_rose: float,
    basse: float,
    haute: float,
) -> float:
    """`w (f2 - f1) + K ln(f2 / f1)` : la DSP integree sur la bande.

    Les deux contributions ne se somment pas de la meme facon : le blanc
    croit comme la **largeur** de bande, le rose comme son **logarithme**.
    C'est ce qui fait qu'elargir la bande vers le haut coute cher en blanc,
    et qu'allonger la mesure coute cher en rose.
    """
    if basse <= 0 or haute <= basse:
        msg = f"bande invalide : {basse} Hz a {haute} Hz"
        raise ValueError(msg)
    blanc = densite_blanche * (haute - basse)
    rose = coefficient_rose * math.log(haute / basse)
    return blanc + rose


def tension_efficace_ramenee_a_l_entree(
    densite_en_tension: float,
    densite_en_courant: float,
    resistance_de_source: float,
) -> float:
    """`racine(v_n^2 + (i_n R_s)^2)` : les deux sources du modele reunies.

    Elles s'ajoutent en **puissance** et non en amplitude, parce qu'elles
    sont decorrelees. Et le poids du bruit en courant croit avec la
    resistance de source -- c'est ce qui decide, sur une datasheet, lequel
    des deux domine.
    """
    if resistance_de_source < 0:
        msg = f"resistance de source negative : {resistance_de_source} ohm"
        raise ValueError(msg)
    return math.hypot(densite_en_tension, densite_en_courant * resistance_de_source)


def facteur_de_bruit(rapport_a_l_entree: float, rapport_a_la_sortie: float) -> float:
    """`F = SNR_entree / SNR_sortie`, qui ne peut pas descendre sous l'unite.

    La planche le dit explicitement : « noise factor should always be
    >= 1 ». Un quadripole n'ameliore jamais un rapport signal sur bruit ;
    au mieux il le preserve.
    """
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
    """`NF = 10 log10 F`, en decibels.

    Dix et non vingt : le facteur de bruit est un rapport de puissances,
    pas d'amplitudes. L'erreur double la figure, et elle est courante.
    """
    if facteur < FACTEUR_MINIMAL_DE_BRUIT:
        msg = f"facteur de bruit sous l'unite : {facteur}"
        raise ValueError(msg)
    return DECIBELS_PAR_DECADE * math.log10(facteur)


def temperature_de_bruit_equivalente(
    facteur: float,
    reference: float = TEMPERATURE_DE_REFERENCE,
) -> float:
    """`T_e = (F - 1) T_0` : le bruit du quadripole vu comme une temperature.

    C'est la forme qui rend la formule de Friis lisible -- les
    temperatures de bruit s'additionnent, ponderees par les gains amont.
    """
    if facteur < FACTEUR_MINIMAL_DE_BRUIT:
        msg = f"facteur de bruit sous l'unite : {facteur}"
        raise ValueError(msg)
    return (facteur - 1) * reference
