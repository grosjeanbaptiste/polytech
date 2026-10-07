"""Ce qu'un programme a menu affiche d'utile : ses resultats, sans ses invites.

Un programme des TP melange a l'ecran le menu, les invites de saisie et
les resultats. Pour comparer deux executions, on ne garde que les
resultats : ils ont une forme reconnaissable, que ce module decrit.
"""

from __future__ import annotations

import re
from typing import Final

ETIQUETTES: Final = "TAB1|TAB2|TAB|T_E|T_P|ETUDIANTS|PROFS|MOYENNES|DOYEN"
"""Les tableaux et listes que les programmes affichent sous la forme `NOM : ...`."""
_RESULTAT: Final = re.compile(
    rf"(?:{ETIQUETTES}) :.*"
    r"|TAB2 au transvasement.*|MIN = .*|POSITION.*|PRESENT|ABSENT"
    r"|[+-] PROF.*|Declaration.*|Fin du programme :"
    r"|\w+ est (?:plein|vide)|Tableau (?:plein|vide)"
    r"|Position invalide|Ajout impossible"
    r"|Code:.*|Id:.*|Erreur \.+|C est vide.*"
)


def releve(sortie: str) -> tuple[str, ...]:
    """Les resultats affiches, dans l'ordre ; tout blanc y vaut un espace."""
    trouves = (trouve.group() for trouve in _RESULTAT.finditer(sortie))
    return tuple(" ".join(texte.split()) for texte in trouves)


def nombre(valeur: float) -> str:
    """Un reel tel que `cout` l'ecrit par defaut : six chiffres significatifs."""
    return f"{valeur:.6g}"


def ligne(etiquette: str, elements: list[str] | tuple[str, ...]) -> str:
    """`NOM : a b c`, la forme de tous les affichages de tableaux."""
    return " ".join((f"{etiquette} :", *elements))
