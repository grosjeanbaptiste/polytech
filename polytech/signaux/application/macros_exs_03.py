"""Macros de la feuille « Chapitre 3 : Systemes ». Prefixe `ExsTrois`."""

from __future__ import annotations

import numpy as np
import sympy as sp

from polytech.signaux.application import linearises_exs_03 as li
from polytech.signaux.application import procede_exs_03 as pr
from polytech.signaux.application import reservoir_exs_03 as re
from polytech.signaux.application.formatage import coordonnees, nombre

_DEBITS = np.linspace(0.0, 1.0, 41)
"""L'abscisse de la caracteristique du reservoir, en m^3/s."""


def _linearises() -> dict[str, str]:
    instable = li.transmittance_du_premier(repos=-sp.Rational(1, 2))
    _, _, sortie = li.repos_du_second()
    (zero,) = li.transmittance_du_second().zeros()
    return {
        "ExsTroisPoleInstable": nombre(float(max(instable.poles())), 2),
        "ExsTroisReposSortie": nombre(float(sortie), 2),
        "ExsTroisZero": nombre(float(zero), 2),
        "ExsTroisGainStatique": nombre(float(li.pente_de_la_caracteristique()), 1),
    }


def _reservoir() -> dict[str, str]:
    gain, constante = re.premier_ordre()
    exacte, linearisee = re.hauteurs_apres_la_variation()
    repos = re.hauteur_de_repos()
    hauteurs = np.array([re.RESERVOIR.hauteur_de_repos(q) for q in _DEBITS[1:]])
    tangente = repos + gain * (_DEBITS - re.DEBIT)
    return {
        "ExsTroisHauteur": nombre(repos, 3),
        "ExsTroisGain": nombre(gain, 3),
        "ExsTroisConstante": nombre(constante, 3),
        "ExsTroisExacte": nombre(exacte, 3),
        "ExsTroisLinearisee": nombre(linearisee, 3),
        "ExsTroisErreur": nombre((exacte - linearisee) * 1e3, 1),
        "ExsTroisCourbe": coordonnees(_DEBITS[1:], hauteurs, 4),
        "ExsTroisTangente": coordonnees(_DEBITS[[10, 40]], tangente[[10, 40]], 4),
        "ExsTroisPointDeRepos": coordonnees(np.array([re.DEBIT]), np.array([repos]), 4),
    }


def _procede() -> dict[str, str]:
    regime = pr.regime()
    statique = pr.transmittance().expression.subs(sp.Symbol("p"), 0)
    return {
        "ExsTroisRegimeSortie": nombre(float(regime["y"]), 0),
        "ExsTroisRegimeTrois": nombre(float(regime["x3"]), 0),
        "ExsTroisRegimeQuatre": nombre(float(regime["x4"]), 0),
        "ExsTroisGainDuProcede": nombre(float(statique), 0),
    }


def macros_des_systemes() -> dict[str, str]:
    """Les macros des exercices 1 a 4 ; les exercices 5 et 6 sont litteraux."""
    return {**_linearises(), **_reservoir(), **_procede()}
