"""Macros du TP 1, questions 1 a 3 : convolution et transformee de Fourier.

Les reponses des questions 1 et 2 sont lues par deconvolution, et non
devinees : le document montre la reponse attendue, puis que la
bibliotheque la retrouve sans la connaitre. La question 3 publie la
prediction et l'estimation sur la meme grille de frequences, et leur
plus grand ecart ; et la valeur que donnerait un echantillonnage naif
des bords, pour montrer ce que coute l'oubli du demi.

Les prefixes sont `TpUn`, `TpDeux` et `TpTrois`.
"""

from __future__ import annotations

import numpy as np

from polytech.signaux.application import enonce_lab_01 as e
from polytech.signaux.application.formatage import coordonnees, nombre
from polytech.signaux.domain.echantillons import convoluer, deconvoluer, peigne, porte
from polytech.signaux.domain.spectre_estime import (
    porte_echantillonnee,
    spectre_estime,
    transformee_de_la_porte,
)

GRILLE_SPECTRALE = np.linspace(0.0, 0.5, 101)
"""Les cinq premiers lobes : jusqu'a cinq fois l'inverse de la duree."""


def _macros_du_un() -> dict[str, str]:
    """La porte, le triangle, et la porte retrouvee par deconvolution."""
    a = porte(e.UN_LARGEUR, e.UN_SUPPORT)
    c = convoluer(a, porte(e.UN_LARGEUR, e.UN_LARGEUR))
    b = deconvoluer(c, a)
    utiles = np.flatnonzero(c)
    return {
        "TpUnLargeur": str(e.UN_LARGEUR),
        "TpUnCrete": nombre(float(c.max()), 0),
        "TpUnInstantDeCrete": str(int(c.argmax())),
        "TpUnDuree": str(utiles.size),
        "TpUnRetrouves": str(int(np.count_nonzero(np.isclose(b, 1.0)))),
        "TpUnCourbe": coordonnees(np.arange(30.0), c[:30], 0),
    }


def _macros_du_deux() -> dict[str, str]:
    """Le peigne, et ses impulsions retrouvees par deconvolution."""
    a = porte(e.DEUX_LARGEUR, e.DEUX_SUPPORT)
    b = peigne(e.DEUX_PERIODE, e.DEUX_IMPULSIONS)
    retrouve = deconvoluer(convoluer(a, b), a)
    instants = np.flatnonzero(np.isclose(retrouve, 1.0))
    return {
        "TpDeuxLargeur": str(e.DEUX_LARGEUR),
        "TpDeuxPeriode": str(e.DEUX_PERIODE),
        "TpDeuxImpulsions": str(e.DEUX_IMPULSIONS),
        "TpDeuxInstants": ";".join(str(int(i)) for i in instants),
        "TpDeuxDuree": str(len(convoluer(a, b))),
    }


def _macros_du_trois() -> dict[str, str]:
    """La prediction, l'estimation, leur ecart, et le prix des bords pleins."""
    echantillons = porte_echantillonnee(e.TROIS_DUREE, e.TROIS_FREQUENCE, e.TROIS_ZEROS)
    estime = spectre_estime(echantillons, e.TROIS_FREQUENCE, GRILLE_SPECTRALE)
    predit = transformee_de_la_porte(e.TROIS_DUREE, GRILLE_SPECTRALE)
    pleins = np.concatenate((np.ones(len(echantillons) - e.TROIS_ZEROS), np.zeros(1)))
    naif = spectre_estime(pleins, e.TROIS_FREQUENCE, np.array([0.0]))[0]
    return {
        "TpTroisDuree": nombre(e.TROIS_DUREE, 0),
        "TpTroisFrequence": nombre(e.TROIS_FREQUENCE, 0),
        "TpTroisZeros": str(e.TROIS_ZEROS),
        "TpTroisEchantillons": str(len(echantillons) - e.TROIS_ZEROS),
        "TpTroisPremierZero": nombre(1 / e.TROIS_DUREE, 2),
        "TpTroisEnZero": nombre(float(estime[0]), 3),
        "TpTroisNaif": nombre(float(naif), 3),
        "TpTroisEcart": nombre(float(np.max(np.abs(estime - predit))), 4),
        "TpTroisPredit": coordonnees(GRILLE_SPECTRALE, predit, 3),
        "TpTroisEstime": coordonnees(GRILLE_SPECTRALE, estime, 3),
    }


def macros_des_signaux() -> dict[str, str]:
    """Les trois premieres questions."""
    return {**_macros_du_un(), **_macros_du_deux(), **_macros_du_trois()}
