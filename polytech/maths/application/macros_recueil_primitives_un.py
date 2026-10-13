r"""Macros du recueil de primitives, exercice 1, prefixe `PrimUn`.

Pour chaque primitive : `Integrande` et `Primitive`. Un changement de
variable y ajoute `Variable`, `Differentielle`, `Transformee` et
`EnVariable` ; les autres gestes, leurs etats intermediaires `EtapeA`,
`EtapeB`, ... Les deux recurrences ont leur terme integre et leurs trois
premiers ordres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.application import recueil_primitives_enonces as enonces
from polytech.maths.application import recueil_primitives_un as un
from polytech.maths.application.formatage import formule
from polytech.maths.application.recueil_algebre import nom
from polytech.maths.application.recueil_primitives_gestes import T
from polytech.maths.application.recueil_primitives_presentation import presenter
from polytech.maths.domain.symboles import VARIABLE as X

if TYPE_CHECKING:
    from polytech.maths.application.resolution_exs_01 import Resolution

PREFIXE = "PrimUn"
CHANGEMENT = "changement de variable"
IMMEDIATE = "primitive immediate"
"""Sa seule etape est la primitive elle-meme : rien de plus a publier."""
ORDRES = {1: "Un", 2: "Deux", 3: "Trois"}


def differentielle(variable: sp.Expr, depuis: sp.Symbol) -> str:
    r"""`g'(x)`, entre parentheses si c'est une somme : elle multiplie `dx`."""
    derivee = sp.diff(variable, depuis)
    if derivee.is_rational_function(depuis):
        derivee = sp.factor(sp.cancel(derivee))
    rendu = formule(derivee)
    return rf"\left({rendu}\right)" if derivee.is_Add else rendu


def _changement(numero: int, resolution: Resolution) -> dict[str, str]:
    variable, transformee, en_variable = resolution.etapes
    depuis = T if variable.has(T) else X
    return {
        nom(PREFIXE, numero, "Variable"): formule(variable),
        nom(PREFIXE, numero, "Differentielle"): differentielle(variable, depuis),
        nom(PREFIXE, numero, "Transformee"): formule(transformee),
        nom(PREFIXE, numero, "EnVariable"): formule(en_variable),
    }


def _etapes(numero: int, resolution: Resolution) -> dict[str, str]:
    if resolution.methode == CHANGEMENT:
        return _changement(numero, resolution)
    if resolution.methode == IMMEDIATE:
        return {}
    lettres = zip("ABCD", resolution.etapes, strict=False)
    return {nom(PREFIXE, numero, f"Etape{c}"): formule(e) for c, e in lettres}


def _recurrences() -> dict[str, str]:
    macros = {
        nom(PREFIXE, 45, "Crochet"): formule(un.relation_45()[0]),
        nom(PREFIXE, 49, "Crochet"): formule(un.relation_49()[0]),
    }
    for ordre, lettres in ORDRES.items():
        logarithme = un.puissance_du_logarithme(ordre)
        exponentielle = sp.factor(un.puissance_fois_exponentielle(ordre))
        macros[nom(PREFIXE, 45, lettres)] = formule(logarithme)
        macros[nom(PREFIXE, 49, lettres)] = formule(exponentielle)
    return macros


def macros_de_l_exercice_un() -> dict[str, str]:
    """Toutes les formules que le corrige de l'exercice 1 compose."""
    macros = _recurrences()
    for numero, integrande in enonces.EXERCICE_UN.items():
        macros[nom(PREFIXE, numero, "Integrande")] = formule(integrande)
    for numero, resolution in un.resolutions().items():
        macros |= _etapes(numero, resolution)
        primitive = presenter(numero, resolution.primitive)
        macros[nom(PREFIXE, numero, "Primitive")] = formule(primitive)
    return macros
