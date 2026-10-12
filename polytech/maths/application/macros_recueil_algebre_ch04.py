r"""Macros du corrige du chapitre IV du recueil d'algebre, prefixe `AlgQuatre`."""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import recueil_algebre_ch04 as chapitre
from polytech.maths.application.recueil_algebre import ecriture, liste, nom

PREFIXE = "AlgQuatre"


def _divisions_et_racines() -> dict[str, str]:
    macros: dict[str, str] = {}
    for lettre, (quotient, reste) in chapitre.exercice_1().items():
        macros[nom(PREFIXE, 1, f"{lettre}Quotient")] = ecriture(quotient)
        macros[nom(PREFIXE, 1, f"{lettre}Reste")] = ecriture(reste)
    for lettre, racines in chapitre.exercice_2().items():
        macros[nom(PREFIXE, 2, lettre)] = liste(sorted(racines))
    for lettre, quotient in chapitre.quotients_du_2().items():
        macros[nom(PREFIXE, 2, f"{lettre}Quotient")] = ecriture(quotient)
    for lettre, (dominant, facteurs) in chapitre.exercice_3().items():
        tete = [] if dominant == 1 else [dominant]
        produit = sp.Mul(*tete, *facteurs, evaluate=False)
        macros[nom(PREFIXE, 3, lettre)] = ecriture(produit)
    return macros


def macros_du_chapitre_quatre() -> dict[str, str]:
    """Toutes les formules que le corrige du chapitre IV compose."""
    quotient, racines = chapitre.exercice_4()
    a, b = chapitre.exercice_5()
    macros = _divisions_et_racines()
    macros[nom(PREFIXE, 4, "Quotient")] = ecriture(quotient)
    macros[nom(PREFIXE, 4, "Racines")] = liste(racines)
    macros[nom(PREFIXE, 5, "a")] = ecriture(a)
    macros[nom(PREFIXE, 5, "b")] = ecriture(b)
    macros |= {
        nom(PREFIXE, 6, k): ecriture(v) for k, v in chapitre.exercice_6().items()
    }
    macros[f"{PREFIXE}CourbeDeuxA"] = chapitre.courbe_du_2a()
    return macros
