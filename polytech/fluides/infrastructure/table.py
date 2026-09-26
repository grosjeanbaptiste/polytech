r"""Adaptateur du port :class:`~fluides.domain.ports.PublicationDeTable`.

Ecrit la table de fleches que ``pgfplots`` lit sous la figure des lignes de
courant. Meme regle que pour les macros : le fichier est genere, jamais
edite a la main.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from polytech.fluides.domain.cinematique.echantillonnage import Fleche

_ENTETE = (
    "% Fichier genere par `python -m polytech.fluides` -- NE PAS EDITER A LA MAIN.\n"
)
_COLONNES = "x y u v\n"
_DECIMALES = 4


@dataclass(frozen=True, slots=True)
class FichierDeTable:
    """Publie les fleches au format table de pgfplots."""

    chemin: Path

    def publier(self, fleches: Sequence[Fleche]) -> None:
        """Ecrit le fichier ; toute erreur d'ecriture se propage."""
        if not fleches:
            msg = "aucune fleche a publier"
            raise ValueError(msg)
        lignes = [_ENTETE, _COLONNES]
        lignes.extend(
            f"{f.x:.{_DECIMALES}f} {f.y:.{_DECIMALES}f} "
            f"{f.u:.{_DECIMALES}f} {f.v:.{_DECIMALES}f}\n"
            for f in fleches
        )
        self.chemin.write_text("".join(lignes), encoding="utf-8")
