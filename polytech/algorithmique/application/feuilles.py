"""Comparer deux feuilles de TP sur leur texte converti, phrase par phrase.

La feuille de 2017 reprend les TP 4 a 6 de celle de 2019. Pour l'etablir
plutot que l'affirmer, on reduit chaque texte a ses **phrases d'enonce**
-- les lignes d'au moins huit mots -- en ecartant ce que la conversion
rend differemment d'un document a l'autre : pieds de page, textes
d'images, et le schema des classes du TP 6.
"""

from __future__ import annotations

import re
from typing import Final

MOTS_D_UNE_PHRASE: Final = 8
DEBUT_DU_TP4: Final = "tableau de structures contenant"
"""Le but du TP 4 : la premiere phrase commune aux deux feuilles."""
_BALISES: Final = re.compile(r"[*#`|>_]|<br")
_PUCE: Final = re.compile(r"^\s*[-•]\s*")
_BRUIT: Final = re.compile(
    r"Faculté Polytechnique|picture text|public :|Vector et List\. Utilisation"
)
"""Pieds de page, textes d'images, et le schema du TP 6 avec le but qu'il coupe."""


def phrases(texte: str) -> tuple[str, ...]:
    """Les phrases d'enonce d'un texte converti, mises a nu."""
    retenues = []
    for brute in texte.splitlines():
        if _BRUIT.search(brute):
            continue
        ligne = " ".join(_PUCE.sub("", _BALISES.sub("", brute)).split())
        if len(ligne.split()) >= MOTS_D_UNE_PHRASE:
            retenues.append(ligne)
    return tuple(retenues)


def a_partir_de(les_phrases: tuple[str, ...], repere: str) -> tuple[str, ...]:
    """Les phrases a partir de la premiere qui contient ce repere."""
    for rang, phrase in enumerate(les_phrases):
        if repere in phrase:
            return les_phrases[rang:]
    msg = f"repere introuvable : {repere}"
    raise ValueError(msg)
