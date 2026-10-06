"""Macros du TP « Reservoir », premiere partie : le modele et la boucle ouverte.

Prefixe `TpCuves`.
"""

from __future__ import annotations

import numpy as np

from polytech.signaux.application import cuves_modele as mo
from polytech.signaux.application.formatage import coordonnees, nombre

_DUREE = 3000.0
"""Dix fois la constante de temps de la premiere cuve, en secondes."""


def _repos() -> dict[str, str]:
    par_le_debit, par_la_concentration = mo.gains_du_melange()
    constantes = mo.constantes_de_temps()
    return {
        "TpCuvesDebit": nombre(mo.debit_auxiliaire(mo.OUVERTURE_DE_REPOS) * 1e3, 3),
        "TpCuvesRepos": nombre(mo.concentration_de_repos(), 3),
        "TpCuvesVanne": nombre(mo.gain_de_la_vanne(), 5),
        "TpCuvesMelange": nombre(par_le_debit, 1),
        "TpCuvesAlimentation": nombre(par_la_concentration, 3),
        "TpCuvesConstanteUn": nombre(constantes[0], 1),
        "TpCuvesConstanteDeux": nombre(constantes[1], 1),
        "TpCuvesConstanteTrois": nombre(constantes[2], 1),
    }


def _nouveau_repos() -> float:
    """La concentration d'equilibre, exacte, une fois la vanne ouverte de dix points."""
    ouverture = mo.OUVERTURE_DE_REPOS + mo.ECHELON_D_OUVERTURE
    return mo.entree_des_cuves(mo.debit_auxiliaire(ouverture), mo.ALIMENTATION_DE_REPOS)


def _boucle_ouverte() -> dict[str, str]:
    commande, perturbation = mo.variations_finales_lineaires()
    instants, ouvert = mo.simuler(
        ouverture=mo.OUVERTURE_DE_REPOS + mo.ECHELON_D_OUVERTURE,
        alimentation=mo.ALIMENTATION_DE_REPOS,
        duree=_DUREE,
    )
    repos = mo.concentration_de_repos()
    pas = slice(None, None, 6)
    macros = {
        "TpCuvesFinaleLineaire": nombre(commande, 3),
        "TpCuvesFinaleReelle": nombre(_nouveau_repos() - repos, 3),
        "TpCuvesFinalePerturbee": nombre(perturbation, 2),
    }
    for rang, nom in enumerate(("Un", "Deux", "Trois")):
        macros[f"TpCuvesOuverte{nom}"] = coordonnees(
            instants[pas], ouvert[pas, rang], 4
        )
    macros["TpCuvesAsymptote"] = coordonnees(
        np.array([0.0, _DUREE]), np.full(2, repos + commande), 4
    )
    return macros


def macros_du_modele() -> dict[str, str]:
    """Les macros du point de repos, de la linearisation et de la boucle ouverte."""
    return {**_repos(), **_boucle_ouverte()}
