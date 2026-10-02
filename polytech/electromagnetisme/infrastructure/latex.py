r"""Adaptateur du port `PublicationDeMacros`.

Ecrit un fichier de `\newcommand` que le document inclut. C'est le seul
endroit de ce contexte qui touche au systeme de fichiers.

Les macros sont **triees** par nom, ce qui rend le fichier
reproductible : deux executions donnent le meme octet, donc un diff ne
montre que ce qui a vraiment change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_ENTETE = (
    "% Fichier genere par `python -m polytech.electromagnetisme` -- "
    "NE PAS EDITER A LA MAIN.\n"
    "% Toute valeur ci-dessous est calculee par la bibliotheque\n"
    "% `electromagnetisme`, adossee au cours et couverte par les tests.\n"
)


@dataclass(frozen=True, slots=True)
class FichierDeMacros:
    """Publie les macros dans un fragment LaTeX que le corrige inclut."""

    chemin: Path

    def publier(self, macros: Mapping[str, str]) -> None:
        """Ecrit le fichier ; toute erreur d'ecriture se propage."""
        if not macros:
            msg = "aucune macro a publier"
            raise ValueError(msg)
        lignes = [_ENTETE]
        for nom, valeur in sorted(macros.items()):
            self._exiger_nom_valide(nom)
            lignes.append(rf"\newcommand{{\{nom}}}{{{valeur}}}" + "\n")
        self.chemin.write_text("".join(lignes), encoding="utf-8")

    @staticmethod
    def _exiger_nom_valide(nom: str) -> None:
        """Un nom de macro LaTeX ne porte que des lettres."""
        if not nom.isalpha():
            msg = f"nom de macro LaTeX invalide (lettres seulement) : {nom!r}"
            raise ValueError(msg)
