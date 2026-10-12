r"""Macros du corrige du chapitre I du recueil d'algebre, prefixe `AlgUn`.

`\AlgUnDixHuitC` est la reponse de l'exercice 18 c) ; un module et un
argument se publient a part, `\AlgUnTroisAModule` et `\AlgUnTroisAArgument`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.maths.application import recueil_algebre_ch01_calculs as calculs
from polytech.maths.application import recueil_algebre_ch01_lieux as lieux
from polytech.maths.application import recueil_algebre_ch01_racines as racines
from polytech.maths.application.recueil_algebre import ecriture, liste, nom, points

if TYPE_CHECKING:
    import sympy as sp

PREFIXE = "AlgUn"


def _polaire(numero: int, cle: str, forme: tuple[sp.Expr, sp.Expr]) -> dict[str, str]:
    return {
        nom(PREFIXE, numero, f"{cle}Module"): ecriture(forme[0]),
        nom(PREFIXE, numero, f"{cle}Argument"): ecriture(forme[1]),
    }


def _calculs() -> dict[str, str]:
    macros = {nom(PREFIXE, 1, k): ecriture(z) for k, z in calculs.exercice_1().items()}
    for lettre, forme in calculs.exercice_3().items():
        macros |= _polaire(3, lettre, forme)
    macros |= _polaire(4, "", calculs.exercice_4())
    macros[nom(PREFIXE, 5)] = ecriture(calculs.exercice_5())
    macros[nom(PREFIXE, 6)] = ecriture(calculs.exercice_6())
    for cle, forme in calculs.exercice_8().items():
        macros |= _polaire(8, cle, forme)
    macros |= {
        nom(PREFIXE, 11, k): ecriture(z) for k, z in calculs.exercice_11().items()
    }
    return macros


def _racines() -> dict[str, str]:
    unites = {3: "Cubiques", 4: "Quatriemes", 6: "Sixiemes"}
    quatorze, quinze = racines.exercice_14(), racines.exercice_15()
    macros = {nom(PREFIXE, 14, unites[n]): liste(zs) for n, zs in quatorze.items()}
    macros |= {nom(PREFIXE, 15, k): liste(zs) for k, zs in quinze.items()}
    macros |= {
        nom(PREFIXE, 18, k): liste(zs) for k, zs in racines.exercice_18().items()
    }
    macros[f"{PREFIXE}PointsSixiemes"] = points(quatorze[6])
    macros[f"{PREFIXE}PointsQuinzeB"] = points(quinze["b"])
    return macros


def _lieux() -> dict[str, str]:
    reelle, imaginaire = lieux.exercice_2()
    macros = {f"{PREFIXE}DeuxReelle": ecriture(reelle)}
    macros[f"{PREFIXE}DeuxImaginaire"] = ecriture(imaginaire.factor())
    macros |= {nom(PREFIXE, 12, k): ecriture(e) for k, e in lieux.exercice_12().items()}
    macros |= {nom(PREFIXE, 13, k): ecriture(e) for k, e in lieux.exercice_13().items()}
    return macros


def macros_du_chapitre_un() -> dict[str, str]:
    """Toutes les formules que le corrige du chapitre I compose."""
    return {**_calculs(), **_racines(), **_lieux()}
