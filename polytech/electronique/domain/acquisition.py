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


def densite_de_bruit_maximale(tension_efficace: float, bande: float) -> float:
    """Densite spectrale que le budget autorise, en volts par racine de hertz."""
    if tension_efficace <= 0 or bande <= 0:
        msg = f"budget ou bande non positif : {tension_efficace}, {bande}"
        raise ValueError(msg)
    return math.sqrt(tension_efficace**2 / bande)
