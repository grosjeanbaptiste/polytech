r"""Adaptateur du port `PublicationDeMacros`.

Ecrit un fichier de `\newcommand` que le document inclut. C'est le seul
endroit de ce contexte qui touche au systeme de fichiers.

Les macros sont **triees** par nom, ce qui rend le fichier
reproductible : deux executions donnent le meme octet, donc un diff ne
montre que ce qui a vraiment change.

Et parce que le fichier est reproductible, l'ecriture est **conditionnee**
a un changement reel de contenu : republier a l'identique ne touche pas
la date de modification. Sans cela, chaque regeneration ferait passer
tous les PDF deja compiles pour perimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_ENTETE = (
    "% Fichier genere par `python -m polytech.signaux` -- "
    "NE PAS EDITER A LA MAIN.\n"
    "% Toute valeur ci-dessous est calculee par la bibliotheque\n"
    "% `signaux`, adossee au cours et couverte par les tests.\n"
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
        self._ecrire_si_different("".join(lignes))

    def _ecrire_si_different(self, contenu: str) -> None:
        """N'ecrit que si le contenu change, pour ne pas bouger la date.

        Sans cette garde, regenerer les macros reecrit chaque fichier a
        l'identique et pousse sa date de modification. La garde « le PDF
        n'est pas anterieur a ses valeurs » tomberait alors pour tous les
        travaux a la fois, alors qu'aucune valeur n'aurait change -- et
        il faudrait recompiler chaque document a chaque passage.

        La date de modification redevient ainsi ce qu'elle pretend etre :
        la date du dernier **changement** de valeur.
        """
        if self.chemin.is_file() and (
            self.chemin.read_text(encoding="utf-8") == contenu
        ):
            return
        self.chemin.write_text(contenu, encoding="utf-8")

    @staticmethod
    def _exiger_nom_valide(nom: str) -> None:
        """Un nom de macro LaTeX ne porte que des lettres."""
        if not nom.isalpha():
            msg = f"nom de macro LaTeX invalide (lettres seulement) : {nom!r}"
            raise ValueError(msg)
