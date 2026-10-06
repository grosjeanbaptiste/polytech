"""Macros de la feuille « Boucle fermee - stabilite ». Prefixe `ExsSept`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from polytech.signaux.application import asservi_exs_07 as un
from polytech.signaux.application import courbes_exs_06 as co
from polytech.signaux.application import nyquist_exs_07 as deux
from polytech.signaux.application import trois_boucles_exs_07 as trois
from polytech.signaux.application.formatage import coordonnees, nombre

if TYPE_CHECKING:
    import sympy as sp

_INSTANTS = np.linspace(0.0, 120.0, 241)
"""Deux minutes : le temps que la boucle a retard se pose."""


def _plus_a_droite(poles: tuple[sp.Expr, ...]) -> float:
    """La plus grande partie reelle : celle qui decide de la stabilite."""
    return max(complex(pole).real for pole in poles)


def _asservi() -> dict[str, str]:
    pulsation, abscisse = un.croisement()
    ouverte = un.boucle().ouverte().subs(un.K, 1)
    instable = _plus_a_droite(un.poles_en_boucle_fermee(1))
    return {
        "ExsSeptUnNyquist": co.nyquist(ouverte, 0.12, 60.0),
        "ExsSeptUnPulsation": nombre(float(pulsation), 3),
        "ExsSeptUnCroisement": nombre(float(abscisse), 2),
        "ExsSeptUnPoint": co.point(float(abscisse), 0.0),
        "ExsSeptUnGainCritique": nombre(float(un.gain_critique_par_nyquist()), 2),
        "ExsSeptUnPoleInstable": nombre(instable, 2),
    }


def _nyquist() -> dict[str, str]:
    rapide, lent = sorted(float(pole) for pole in deux.poles_en_boucle_fermee(20))
    return {
        "ExsSeptDeuxNyquist": co.nyquist(deux.TRANSMITTANCE, 0.03, 60.0),
        "ExsSeptDeuxVerticale": co.verticale(float(deux.asymptote()), -14.0, 1.0),
        "ExsSeptDeuxPoleLent": nombre(lent, 2),
        "ExsSeptDeuxPoleRapide": nombre(rapide, 2),
    }


def _trois_boucles() -> dict[str, str]:
    reponse = trois.indicielle_de_b(_INSTANTS)
    instable = _plus_a_droite(trois.poles_en_boucle_fermee("a"))
    return {
        "ExsSeptTroisIndicielle": coordonnees(_INSTANTS, reponse, 4),
        "ExsSeptTroisPoleInstable": nombre(instable, 2),
        "ExsSeptTroisModule": nombre(trois.module_de_b(1.0), 3),
    }


def macros_de_la_stabilite() -> dict[str, str]:
    """Les macros des trois exercices."""
    return {**_asservi(), **_nyquist(), **_trois_boucles()}
