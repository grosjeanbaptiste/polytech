"""Laboratoire 3 d'electronique, exercice 1.2, fin : le deuxieme bloc et le total.

Apres l'AMP3 de tete, pousse a son gain maximal, tout bloc suivant voit
son bruit divise par le carre de ce gain. Le module chiffre la question 6,
puis ferme le budget sur la chaine entiere -- ce que l'enonce ne demande
pas.
"""

from __future__ import annotations

import math

from polytech.electronique.application import chaine_1_2_lab_03 as amont
from polytech.electronique.application.donnees_lab_03 import UnDeux as D
from polytech.electronique.domain.chaine_d_acquisition import (
    Etage,
    puissances_a_l_entree,
)


def bruit_du_second_amplificateur(nom: str) -> float:
    """Son bruit rapporte a l'entree, derriere le gain du premier."""
    return amont.bruits_des_amplificateurs()[nom] / amont.gain_maximal_du_premier() ** 2


def bruit_du_filtre_en_second() -> float:
    """Celui du filtre s'il vient juste apres le premier amplificateur."""
    return (
        amont.bruit_d_un_bloc_en_tete(D.FILTRE) / amont.gain_maximal_du_premier() ** 2
    )


def gain_du_second() -> float:
    """Ce qui manque pour atteindre le gain total."""
    return amont.gain() / amont.gain_maximal_du_premier()


def puissances_de_la_chaine() -> dict[str, float]:
    """AMP3, AMP1, filtre, M2 : la puissance de chacun rapportee a l'entree."""
    racine = math.sqrt(D.BANDE)
    blocs = (
        Etage(
            "AMP3",
            amont.gain_maximal_du_premier(),
            D.AMPLIFICATEURS["AMP3"][0] / racine,
        ),
        Etage("AMP1", gain_du_second(), D.AMPLIFICATEURS["AMP1"][0] / racine),
        Etage("filtre", 1.0, D.FILTRE),
        Etage(
            amont.multiplexeur_retenu(),
            1.0,
            D.MULTIPLEXEURS[amont.multiplexeur_retenu()][0],
        ),
    )
    valeurs = puissances_a_l_entree(blocs, D.BANDE)
    return {bloc.nom: valeur for bloc, valeur in zip(blocs, valeurs, strict=True)}


def bruit_de_la_chaine() -> float:
    """La tension efficace de bruit de la chaine entiere, a l'entree."""
    return math.sqrt(sum(puissances_de_la_chaine().values()))


def cout() -> float:
    """Le prix des composants dont l'enonce donne le prix, en euros."""
    par_voie = D.AMPLIFICATEURS["AMP3"][3] + D.AMPLIFICATEURS["AMP1"][3]
    return D.VOIES * par_voie + D.MULTIPLEXEURS[amont.multiplexeur_retenu()][1]
