r"""Macros du corrige du chapitre VIII, exercices 1 a 13, prefixe `AlgHuit`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.maths.application import recueil_algebre_ch08a as partie
from polytech.maths.application import recueil_algebre_ch08a_suite as suite
from polytech.maths.application import recueil_algebre_espaces as espaces
from polytech.maths.application.recueil_algebre import ecriture, nom

if TYPE_CHECKING:
    from collections.abc import Sequence

PREFIXE = "AlgHuit"
NOMS_DU_13 = {-1: "MoinsUn", 1: "Un"}


def _sommets(points: Sequence[Sequence[object]]) -> str:
    return " ".join(f"({x},{y})" for x, y in points)


def _premiers() -> dict[str, str]:
    macros: dict[str, str] = {}
    for lettre, (premiere, seconde) in partie.exercice_2().items():
        macros[nom(PREFIXE, 2, f"{lettre}Un")] = espaces.vecteur(premiere)
        macros[nom(PREFIXE, 2, f"{lettre}Deux")] = espaces.vecteur(seconde)
    macros[nom(PREFIXE, 5, "aNoyau")] = espaces.famille(
        partie.APPLICATIONS["5a"].noyau()
    )
    macros[nom(PREFIXE, 6)] = espaces.relation(partie.exercice_6()[1], "X")
    macros[nom(PREFIXE, 7)] = espaces.vecteur(partie.exercice_7())
    macros[nom(PREFIXE, 8)] = ecriture(partie.noyau_de_la_derivation_moins(partie.A))
    for lettre, determinant in partie.determinants_du_9().items():
        macros[nom(PREFIXE, 9, f"{lettre}Det")] = ecriture(determinant)
    for lettre, inverse in partie.exercice_9().items():
        if inverse is not None:
            macros[nom(PREFIXE, 9, lettre)] = espaces.vecteur(inverse)
    return macros


def macros_de_la_partie_a() -> dict[str, str]:
    """Toutes les formules que le corrige des exercices 1 a 13 compose."""
    carre, image = partie.carre_et_son_image()
    macros = _premiers()
    macros[f"{PREFIXE}Carre"] = _sommets(carre)
    macros[f"{PREFIXE}CarreImage"] = _sommets(image)
    macros[nom(PREFIXE, 13, "Det")] = ecriture(suite.MATRICE_13.det().factor())
    for valeur, base in suite.images_du_13().items():
        macros[nom(PREFIXE, 13, NOMS_DU_13[valeur])] = espaces.famille(base)
    for valeur, base in suite.completees_du_13().items():
        cle = f"{NOMS_DU_13[valeur]}Completee"
        macros[nom(PREFIXE, 13, cle)] = espaces.famille(base)
    return macros
