r"""Macros du corrige du chapitre III du recueil d'algebre, prefixe `AlgTrois`.

Pour un systeme sans parametre : `Operations`, `Echelonnee`, puis `Simple`
et `Solution` s'il est compatible. Pour un systeme a discuter : `Det`,
`Generique`, et par cas `<Cas>Forme` et `<Cas>Solution`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.maths.application import recueil_algebre_ch03_enonces as enonces
from polytech.maths.application import recueil_algebre_ch03_resolution as resolution
from polytech.maths.application.recueil_algebre import ecriture, liste, nom, operations
from polytech.maths.domain.algebre.echelonnement import echelonner
from polytech.maths.domain.algebre.solutions import Nature

if TYPE_CHECKING:
    import sympy as sp

PREFIXE = "AlgTrois"


def _uplet(solution: tuple[sp.Expr, ...]) -> str:
    r"""`\left(x,\; y,\; z\right)` : des virgules, qu'une matrice ligne n'a pas."""
    return rf"\left({liste(solution)}\right)"


def _sans_parametre(cle: str, augmentee: sp.Matrix) -> dict[str, str]:
    numero = int(cle.rstrip("ab"))
    lettre = cle.removeprefix(str(numero))
    echelonnee = echelonner(augmentee, colonnes=augmentee.cols - 1)
    macros = {
        nom(PREFIXE, numero, f"{lettre}Operations"): operations(echelonnee.operations),
        nom(PREFIXE, numero, f"{lettre}Echelonnee"): ecriture(echelonnee.matrice),
    }
    if resolution.nature(augmentee) is not Nature.IMPOSSIBLE:
        simple, generale = augmentee.rref()[0], resolution.solution(augmentee)
        macros[nom(PREFIXE, numero, f"{lettre}Simple")] = ecriture(simple)
        macros[nom(PREFIXE, numero, f"{lettre}Solution")] = _uplet(generale)
    return macros


def _a_discuter(numero: int) -> dict[str, str]:
    macros: dict[str, str] = {}
    for cle in enonces.FORMES[numero]:
        forme = resolution.forme_du_cas(numero, cle)
        macros[nom(PREFIXE, numero, f"{cle}Forme")] = ecriture(forme)
    for cle in enonces.SOLUTIONS[numero]:
        generale = resolution.solution_du_cas(numero, cle)
        macros[nom(PREFIXE, numero, f"{cle}Solution")] = _uplet(generale)
    augmentee = enonces.PARAMETRIQUES[numero]
    if augmentee.rows == augmentee.cols - 1:
        macros[nom(PREFIXE, numero, "Det")] = ecriture(resolution.determinant(numero))
    if numero in enonces.A_SOLUTION_UNIQUE:
        unique = resolution.generique(numero)
        macros[nom(PREFIXE, numero, "Generique")] = _uplet(unique)
    return macros


def macros_du_chapitre_trois() -> dict[str, str]:
    """Toutes les formules que le corrige du chapitre III compose."""
    macros: dict[str, str] = {}
    for cle, augmentee in enonces.NUMERIQUES.items():
        macros |= _sans_parametre(cle, augmentee)
    for numero in enonces.PARAMETRIQUES:
        macros |= _a_discuter(numero)
    return macros
