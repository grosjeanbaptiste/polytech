"""L'effet pelliculaire : l'epaisseur de peau, et ce qu'elle coute vraiment.

`delta = racine(2 / (omega mu sigma))`. La note du cours qui lui est
consacree insiste sur une consequence pratique : a 50 Hz dans le cuivre,
l'epaisseur de peau vaut environ `9 mm`, ce qui est **du meme ordre** que
le rayon des conducteurs industriels -- donc l'effet n'est pas une
curiosite de haute frequence, il dimensionne les jeux de barres.

La resistance apparente croit comme la racine de la frequence, et c'est ce
rapport qui est expose : il dit directement combien coute une montee en
frequence.
"""

from __future__ import annotations

import math

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE

CONDUCTIVITE_DU_CUIVRE = 5.96e7
"""`sigma` du cuivre a 20 degres, en S/m. Valeur de la litterature."""


def epaisseur_de_peau(
    frequence: float,
    conductivite: float = CONDUCTIVITE_DU_CUIVRE,
    relative: float = 1.0,
) -> float:
    """`delta = racine(2 / (omega mu sigma))`, en metres."""
    if frequence <= 0:
        msg = f"frequence nulle ou negative : {frequence} Hz"
        raise ValueError(msg)
    if conductivite <= 0:
        msg = f"conductivite nulle ou negative : {conductivite} S/m"
        raise ValueError(msg)
    pulsation = 2 * math.pi * frequence
    permeabilite = relative * PERMEABILITE_DU_VIDE
    return math.sqrt(2.0 / (pulsation * permeabilite * conductivite))


def rapport_des_epaisseurs(basse: float, haute: float) -> float:
    """`delta(f1) / delta(f2) = racine(f2 / f1)` : la loi en racine.

    Le rapport est rendu sans calculer les deux epaisseurs, parce que
    c'est lui que le dimensionnement utilise -- et parce qu'il ne depend
    ni du materiau ni de la permeabilite.
    """
    if basse <= 0 or haute <= 0:
        msg = f"frequences nulles ou negatives : {basse} Hz et {haute} Hz"
        raise ValueError(msg)
    return math.sqrt(haute / basse)


def est_negligeable(frequence: float, rayon: float, **materiau: float) -> bool:
    """Vrai si l'epaisseur de peau depasse largement le rayon du conducteur.

    « Largement » veut dire un facteur trois : au-dela, le courant occupe
    toute la section et la resistance continue suffit. En dessous, la
    note du cours montre que l'erreur devient significative.
    """
    if rayon <= 0:
        msg = f"rayon nul ou negatif : {rayon} m"
        raise ValueError(msg)
    facteur = 3.0
    return epaisseur_de_peau(frequence, **materiau) > facteur * rayon
