r"""Macros du corrige du chapitre VII du recueil d'algebre, prefixe `AlgSept`.

Une base s'ecrit dans l'espace de son exercice : des uplets jusqu'au 22,
des matrices du 23 au 29, des polynomes du 31 au 38.
"""

from __future__ import annotations

import re
from itertools import takewhile
from typing import TYPE_CHECKING

from polytech.maths.application import recueil_algebre_ch07_bases as bases
from polytech.maths.application import recueil_algebre_ch07_familles as familles
from polytech.maths.application import recueil_algebre_espaces as espaces
from polytech.maths.application.recueil_algebre import (
    ecriture,
    en_lettres,
    liste,
    nom,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

PREFIXE = "AlgSept"
PREMIER_DES_MATRICES, PREMIER_DES_POLYNOMES, PREMIER_ABSTRAIT = 23, 30, 39
LETTRES = {
    "25": "M",
    "23": "M",
    "26": "M",
    "31": "p",
    "32F1": "p",
    "35": "p",
    "37": "p",
}
"""La lettre qui nomme les vecteurs d'une relation ; `X` par defaut."""


def _decoupe(cle: str) -> tuple[int, str]:
    """`36W1` : l'exercice 36, et `WUn` -- un nom de macro n'a pas de chiffre."""
    numero = int("".join(takewhile(str.isdigit, cle)))
    reste = cle.removeprefix(str(numero))
    return numero, re.sub(r"\d+", lambda m: en_lettres(int(m.group())), reste)


def _forme(numero: int) -> Callable[[Sequence[object]], str]:
    if PREMIER_DES_MATRICES <= numero < PREMIER_DES_POLYNOMES:
        return espaces.matrice
    if PREMIER_DES_POLYNOMES <= numero < PREMIER_ABSTRAIT:
        return espaces.polynome
    return espaces.vecteur


def _famille(
    cle: str, vecteurs: Sequence[Sequence[object]], suffixe: str = ""
) -> dict[str, str]:
    numero, reste = _decoupe(cle)
    return {
        nom(PREFIXE, numero, reste + suffixe): espaces.famille(vecteurs, _forme(numero))
    }


def _familles() -> dict[str, str]:
    macros = {nom(PREFIXE, 5): ecriture(familles.condition_du_5())}
    macros[nom(PREFIXE, 8)] = str(familles.exercice_8())
    for cle, valeurs in familles.coordonnees_demandees().items():
        numero, reste = _decoupe(cle)
        macros[nom(PREFIXE, numero, reste)] = liste(valeurs)
    for cle, coefficients in familles.relations().items():
        numero, reste = _decoupe(cle)
        lettre = LETTRES.get(cle, "X")
        macros[nom(PREFIXE, numero, f"{reste}Relation")] = espaces.relation(
            coefficients, lettre
        )
    for cle, determinant in familles.determinants().items():
        numero, reste = _decoupe(cle)
        macros[nom(PREFIXE, numero, f"{reste}Det")] = ecriture(determinant)
    for cle, nature in familles.natures().items():
        macros |= _famille(cle, nature.completee or nature.extraite, "Base")
    return macros


def _bases() -> dict[str, str]:
    macros: dict[str, str] = {}
    for cle, base in bases.toutes().items():
        macros |= _famille(cle, base)
    for cle, base in bases.prolongees().items():
        macros |= _famille(cle, base, "Prolongee")
    for cle, quadruplet in bases.dimensions().items():
        macros[nom(PREFIXE, int(cle))] = liste(quadruplet)
    (equation,) = bases.exercice_22()
    macros[nom(PREFIXE, 22)] = espaces.equation(equation, "xyzt")
    macros[nom(PREFIXE, 33)] = ecriture(bases.exercice_33())
    return macros


def macros_du_chapitre_sept() -> dict[str, str]:
    """Toutes les formules que le corrige du chapitre VII compose."""
    return {**_familles(), **_bases()}
