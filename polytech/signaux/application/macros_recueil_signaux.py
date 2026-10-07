"""Macros du recueil d'exercices, planches sur les signaux. Prefixe `Recueil`."""

from __future__ import annotations

import numpy as np
import sympy as sp

from polytech.signaux.application import recueil_appariement as ap
from polytech.signaux.application import recueil_convolutions as cv
from polytech.signaux.application import recueil_spectres as sx
from polytech.signaux.application.formatage import coordonnees, nombre

_HORIZON = 20.0
"""La duree portee sur les quatre courbes de la planche 1."""
_UN_SUR = 25
"""Un point sur vingt-cinq suffit a tracer ces courbes."""
_RAIES = 20
"""Le rang de la derniere raie portee de chaque cote, sur la planche 3."""


def _appariement() -> dict[str, str]:
    instants = np.arange(0.0, ap.DUREE, ap.PAS)
    garde = instants < _HORIZON
    noms = {("x1", "h1"): "Un", ("x2", "h1"): "Deux", ("x1", "h2"): "Trois"}
    noms[("x2", "h2")] = "Quatre"
    return {
        f"RecueilUnY{nom}": coordonnees(
            instants[garde][::_UN_SUR], ap.sortie(*couple)[garde][::_UN_SUR], 3
        )
        for couple, nom in noms.items()
    }


def _spectres() -> dict[str, str]:
    rapports = np.linspace(-2.0, 2.0, 401)
    periode = float(sx.REPETITION)
    rangs = np.arange(-_RAIES, _RAIES + 1)
    raies = np.array(sx.modules_de_la_rampe(list(rangs / periode))) / periode
    frequences = np.linspace(-2.0, 2.0, 81)
    enveloppe = np.array(sx.modules_de_la_rampe(list(frequences))) / periode
    return {
        "RecueilDeuxSpectre": coordonnees(
            rapports, np.array(sx.spectre_normalise(list(rapports))), 3
        ),
        "RecueilDeuxPeriodes": str(sx.PERIODES),
        "RecueilDeuxZero": nombre(1 / sx.PERIODES, 3),
        "RecueilTroisNiveau": nombre(sx.niveau_en_decibels(0), 2),
        "RecueilTroisDix": nombre(sx.niveau_en_decibels(10), 1),
        "RecueilTroisVingt": nombre(sx.niveau_en_decibels(20), 1),
        "RecueilTroisRaie": nombre(sx.niveau_de_la_raie(0), 2),
        "RecueilTroisRaies": coordonnees(rangs / periode, raies, 4),
        "RecueilTroisEnveloppe": coordonnees(frequences, enveloppe, 4),
    }


def _trapeze(donnees: cv.Creneaux) -> str:
    sommets = np.array(cv.sommets(donnees), dtype=float)
    return coordonnees(sommets[:, 0], sommets[:, 1], 1)


def macros_du_recueil_signaux() -> dict[str, str]:
    """Les macros des planches 1, 2, 3, 13, 17 et 18."""
    periode = sx.periode_pour_une_puissance_unite()
    return {
        **_appariement(),
        **_spectres(),
        "RecueilTreizeTrapeze": _trapeze(cv.PLANCHE_13),
        "RecueilDixHuitTrapeze": _trapeze(cv.PLANCHE_18),
        "RecueilDixSeptPuissance": sp.latex(sx.puissance(sp.Rational(3, 100))),
        "RecueilDixSeptPeriode": nombre(float(periode) * 1e3, 1),
    }
