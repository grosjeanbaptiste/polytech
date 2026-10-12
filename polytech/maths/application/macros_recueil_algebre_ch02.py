r"""Macros du corrige du chapitre II du recueil d'algebre, prefixe `AlgDeux`."""

from __future__ import annotations

from polytech.maths.application import recueil_algebre_ch02_equations as equations
from polytech.maths.application import recueil_algebre_ch02_formes as formes
from polytech.maths.application.recueil_algebre import (
    ecriture,
    liste,
    nom,
    operations,
)

PREFIXE = "AlgDeux"
ORDRE_MONTRE = 4
"""L'ordre auquel le corrige montre les matrices 10 e) et 10 f)."""


def _equations() -> dict[str, str]:
    hors_diagonale, racines_5 = equations.exercice_5()
    symetrique, antisymetrique = equations.exercice_6()
    determinant, racines_8 = equations.exercice_8()
    return {
        nom(PREFIXE, 1): ecriture(equations.exercice_1()),
        nom(PREFIXE, 2): ecriture(equations.exercice_2()),
        nom(PREFIXE, 4): ecriture(equations.exercice_4()),
        nom(PREFIXE, 5, "Terme"): ecriture(hors_diagonale),
        nom(PREFIXE, 5, "Racines"): liste(racines_5),
        nom(PREFIXE, 6, "Symetrique"): ecriture(symetrique),
        nom(PREFIXE, 6, "Antisymetrique"): ecriture(antisymetrique),
        nom(PREFIXE, 8, "Determinant"): ecriture(determinant),
        nom(PREFIXE, 8, "Racines"): liste(racines_8),
    }


def _formes() -> dict[str, str]:
    macros = {
        nom(PREFIXE, 9, f"{k}Echelonnee"): ecriture(m)
        for k, m in formes.echelonnees_du_9().items()
    }
    macros |= {nom(PREFIXE, 9, k): ecriture(m) for k, m in formes.exercice_9().items()}
    macros |= {
        nom(PREFIXE, 10, k): ecriture(m) for k, m in formes.exercice_10().items()
    }
    macros |= {
        nom(PREFIXE, 9, f"{k}Operations"): operations(journal)
        for k, journal in formes.journaux_du_9().items()
    }
    bidiagonale = formes.inverse_de_la_bidiagonale(ORDRE_MONTRE)
    antidiagonale, inverse = formes.antidiagonale_et_son_inverse(ORDRE_MONTRE)
    macros[nom(PREFIXE, 10, "eQuatre")] = ecriture(bidiagonale)
    macros[nom(PREFIXE, 10, "fMatrice")] = ecriture(antidiagonale)
    macros[nom(PREFIXE, 10, "fInverse")] = ecriture(inverse)
    return macros


def macros_du_chapitre_deux() -> dict[str, str]:
    """Toutes les formules que le corrige du chapitre II compose."""
    return {**_equations(), **_formes()}
