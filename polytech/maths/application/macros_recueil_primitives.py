r"""Macros du recueil de primitives, exercices 2 a 11 : quatre jeux, un par dossier.

Meme serie que pour l'exercice 1 : `Integrande` et `Primitive` ; un
changement de variable y ajoute `Variable`, `Differentielle`,
`Transformee` et `EnVariable` ; les autres gestes leurs etats `EtapeA`,
`EtapeB`, ... Le premier jeu porte en plus la recurrence de l'exercice 3.
"""

from __future__ import annotations

from polytech.maths.application import recueil_primitives_enonces as enonces
from polytech.maths.application import recueil_primitives_recurrence as recurrence
from polytech.maths.application import recueil_primitives_suite as suite
from polytech.maths.application.formatage import formule
from polytech.maths.application.macros_recueil_primitives_un import (
    CHANGEMENT,
    IMMEDIATE,
    differentielle,
)
from polytech.maths.application.recueil_algebre import liste, nom_de_la_cle
from polytech.maths.application.recueil_primitives_gestes import T
from polytech.maths.application.recueil_primitives_presentation import presenter
from polytech.maths.domain.symboles import VARIABLE as X

JEUX = {
    "PrimDeux": ("2", "4"),
    "PrimTrois": ("5", "6", "7"),
    "PrimQuatre": ("8",),
    "PrimCinq": ("9", "10", "11"),
}
"""Prefixe du jeu : les exercices qu'il couvre."""
ORDRES = {1: "Un", 2: "Deux", 3: "Trois"}


def _exercice(cle: str) -> str:
    return "".join(c for c in cle if c.isdigit())


def _etapes(prefixe: str, cle: str) -> dict[str, str]:
    resolution = suite.resolutions()[cle]
    if resolution.methode == IMMEDIATE:
        return {}
    if resolution.methode != CHANGEMENT:
        lettres = zip("ABCD", resolution.etapes, strict=False)
        return {
            nom_de_la_cle(prefixe, cle, f"Etape{c}"): formule(e) for c, e in lettres
        }
    variable, transformee, en_variable = resolution.etapes
    depuis = T if variable.has(T) else X
    return {
        nom_de_la_cle(prefixe, cle, "Variable"): formule(variable),
        nom_de_la_cle(prefixe, cle, "Differentielle"): differentielle(variable, depuis),
        nom_de_la_cle(prefixe, cle, "Transformee"): formule(transformee),
        nom_de_la_cle(prefixe, cle, "EnVariable"): formule(en_variable),
    }


def macros_du_jeu(prefixe: str) -> dict[str, str]:
    """Les formules des primitives des exercices que ce jeu couvre."""
    macros: dict[str, str] = {}
    for cle, integrande in enonces.EXERCICES_SUIVANTS.items():
        if _exercice(cle) not in JEUX[prefixe]:
            continue
        primitive = presenter(cle, suite.resolutions()[cle].primitive)
        macros[nom_de_la_cle(prefixe, cle, "Integrande")] = formule(integrande)
        macros[nom_de_la_cle(prefixe, cle, "Primitive")] = formule(primitive)
        macros |= _etapes(prefixe, cle)
    return macros


def _recurrence(prefixe: str) -> dict[str, str]:
    facteur, terme = recurrence.relation()
    macros = {
        nom_de_la_cle(prefixe, "3Facteur"): formule(facteur),
        nom_de_la_cle(prefixe, "3Terme"): formule(terme),
        nom_de_la_cle(prefixe, "3Centre"): formule(recurrence.CENTRE),
    }
    for ordre, lettres in ORDRES.items():
        coefficients = recurrence.coefficients_du_trinome(ordre)
        devant = recurrence.facteur_du_trinome(ordre)
        macros |= {
            nom_de_la_cle(prefixe, f"3Ordre{lettres}"): formule(
                recurrence.puissance(ordre)
            ),
            nom_de_la_cle(prefixe, f"3Trinome{lettres}"): formule(
                recurrence.forme_du_trinome(ordre)
            ),
            nom_de_la_cle(prefixe, f"3Devant{lettres}"): formule(devant),
            nom_de_la_cle(prefixe, f"3Coefficients{lettres}"): liste(coefficients),
        }
    return macros


def macros_des_exercices_2_a_4() -> dict[str, str]:
    """Fractions rationnelles, recurrence, exponentielles."""
    return macros_du_jeu("PrimDeux") | _recurrence("PrimDeux")


def macros_des_exercices_5_a_7() -> dict[str, str]:
    """Les trois exercices de fonctions diverses."""
    return macros_du_jeu("PrimTrois")


def macros_de_l_exercice_8() -> dict[str, str]:
    """Les rationnelles en sinus et cosinus."""
    return macros_du_jeu("PrimQuatre")


def macros_des_exercices_9_a_11() -> dict[str, str]:
    """Les irrationnelles et la revision."""
    return macros_du_jeu("PrimCinq")
