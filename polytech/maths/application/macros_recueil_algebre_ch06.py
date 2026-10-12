r"""Macros du corrige du chapitre VI du recueil d'algebre, prefixe `AlgSix`.

L'exercice est unique : les macros portent la lettre de la matrice,
`\AlgSixARang`, `\AlgSixCDet`, `\AlgSixCUnForme`.
"""

from __future__ import annotations

from polytech.maths.application import recueil_algebre_ch06 as chapitre
from polytech.maths.application.recueil_algebre import ecriture, operations

PREFIXE = "AlgSix"


def _cle(texte: str) -> str:
    return f"{PREFIXE}{texte[:1].upper()}{texte[1:]}"


def macros_du_chapitre_six() -> dict[str, str]:
    """Toutes les formules que le corrige du chapitre VI compose."""
    macros: dict[str, str] = {}
    rangs = chapitre.rangs_numeriques()
    for lettre, echelonnee in chapitre.echelonnees().items():
        macros[_cle(f"{lettre}Operations")] = operations(echelonnee.operations)
        macros[_cle(f"{lettre}Forme")] = ecriture(echelonnee.matrice)
        macros[_cle(f"{lettre}Rang")] = str(rangs[lettre])
    for lettre, determinant in chapitre.determinants().items():
        macros[_cle(f"{lettre}Det")] = ecriture(determinant)
    for cle, forme in chapitre.formes_singulieres().items():
        macros[_cle(f"{cle}Forme")] = ecriture(forme)
    return macros
