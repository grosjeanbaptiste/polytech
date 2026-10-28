"""Le budget d'une chaine d'acquisition : gain, cadence, resolution, bruit.

Quatre contraintes se rencontrent dans une meme chaine et se contredisent
volontiers. Les tenir separees permet de dire laquelle bloque -- c'est ce
qui distingue « le cahier des charges est impossible » d'un simple echec.
"""

from __future__ import annotations

import math

FACTEUR_DE_NYQUIST = 2
"""Deux echantillons par periode : le minimum pour ne rien replier."""


def gain_de_chaine(tension_visee: float, tension_capteur: float) -> float:
    """Gain qui porte la sortie du capteur a la pleine echelle du convertisseur."""
    if tension_capteur <= 0 or tension_visee <= 0:
        msg = f"tensions non positives : {tension_visee}, {tension_capteur}"
        raise ValueError(msg)
    return tension_visee / tension_capteur


def nyquist(frequence_maximale: float) -> float:
    """Cadence minimale d'echantillonnage d'un signal de cette bande."""
    if frequence_maximale <= 0:
        msg = f"frequence maximale non positive : {frequence_maximale}"
        raise ValueError(msg)
    return FACTEUR_DE_NYQUIST * frequence_maximale


def cadence_du_multiplexeur(cadence_par_voie: float, voies: int) -> float:
    """Le multiplexeur sert chaque voie a son tour : les cadences s'ajoutent."""
    if voies <= 0:
        msg = f"nombre de voies non positif : {voies}"
        raise ValueError(msg)
    return cadence_par_voie * voies


def bits_necessaires(etendue: float, pas_maximal: float) -> int:
    """Resolution minimale pour que le pas de quantification tienne la borne."""
    if etendue <= 0 or pas_maximal <= 0:
        msg = f"etendue ou pas non positif : {etendue}, {pas_maximal}"
        raise ValueError(msg)
    return math.ceil(math.log2(etendue / pas_maximal))


def courant_de_bruit_integre(densite_de_courant: float, bande: float) -> float:
    """`i_n sqrt(B)` : le courant efficace de bruit sur la bande, en ampere."""
    if bande <= 0:
        msg = f"bande non positive : {bande}"
        raise ValueError(msg)
    return densite_de_courant * math.sqrt(bande)


def puissance_de_bruit_en_courant(
    densite_de_courant: float,
    impedance: float,
    bande: float,
) -> float:
    """Puissance que le bruit en courant depose dans l'impedance de source.

    Le courant de bruit traverse l'impedance et s'y convertit en tension :
    c'est ce carre-la qui se compare au budget, pas le courant lui-meme.
    """
    if bande <= 0:
        msg = f"bande non positive : {bande}"
        raise ValueError(msg)
    return (densite_de_courant * impedance) ** 2 * bande


def tension_de_bruit_en_courant(
    densite_de_courant: float, impedance: float, bande: float
) -> float:
    """La tension efficace que ce bruit en courant fait aux bornes de l'impedance."""
    return math.sqrt(
        puissance_de_bruit_en_courant(densite_de_courant, impedance, bande)
    )


def densite_de_bruit_maximale(tension_efficace: float, bande: float) -> float:
    """Densite spectrale que le budget autorise, en volts par racine de hertz."""
    if tension_efficace <= 0 or bande <= 0:
        msg = f"budget ou bande non positif : {tension_efficace}, {bande}"
        raise ValueError(msg)
    return math.sqrt(tension_efficace**2 / bande)


FACTEUR_UNILATERAL_BILATERAL = 2
"""La convention unilaterale ne compte que les frequences positives.

Elle vaut donc le double de la bilaterale. Le fascicule consacre une
planche a la distinction parce que la confondre double ou divise par deux
toute puissance de bruit calculee.
"""


def autocorrelation_d_un_bruit_blanc(densite: float, retard: float) -> float:
    """`S delta(tau)` : nulle partout sauf a l'origine.

    C'est la definition meme de la blancheur, et ce qui rend la densite
    spectrale plate. Deux echantillons distincts sont decorreles, quelle
    que soit leur proximite -- idealisation, puisqu'aucun bruit reel n'a de
    bande infinie.
    """
    if densite < 0:
        msg = f"densite spectrale negative : {densite}"
        raise ValueError(msg)
    return densite if retard == 0 else 0.0


def autocovariance(correlation: float, moyenne: float) -> float:
    """`C(tau) = R(tau) - m^2` : les deux coincident pour un signal centre."""
    return correlation - moyenne**2


def densite_spectrale_bilaterale(unilaterale: float) -> float:
    """La bilaterale vaut la moitie de l'unilaterale."""
    if unilaterale < 0:
        msg = f"densite spectrale negative : {unilaterale}"
        raise ValueError(msg)
    return unilaterale / FACTEUR_UNILATERAL_BILATERAL


def densite_spectrale_unilaterale_depuis_bilaterale(bilaterale: float) -> float:
    """Et reciproquement : l'unilaterale vaut le double."""
    if bilaterale < 0:
        msg = f"densite spectrale negative : {bilaterale}"
        raise ValueError(msg)
    return bilaterale * FACTEUR_UNILATERAL_BILATERAL


def variance_d_un_bruit_borne(densite: float, bande: float) -> float:
    """`sigma^2 = S B` : la puissance d'un bruit blanc limite en bande.

    Doubler la bande double la puissance -- c'est pourquoi on filtre avant
    de mesurer, et non apres.
    """
    if bande <= 0:
        msg = f"bande nulle ou negative : {bande} Hz"
        raise ValueError(msg)
    if densite < 0:
        msg = f"densite spectrale negative : {densite}"
        raise ValueError(msg)
    return densite * bande
