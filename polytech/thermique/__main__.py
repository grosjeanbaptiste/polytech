"""Point d'entree : regenere les macros que le corrige du TP 1 consomme.

Usage : ``python -m polytech.thermique [chemin]``. Avec un chemin, seul le
TP 1 est regenere, vers ce chemin. Sans argument, le TP 1 l'est vers son
dossier, puis chaque travail inscrit au registre vers le sien.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from polytech.thermique.application.generation import GenererMacros
from polytech.thermique.application.generation_du_travail import (
    GenererMacrosDuTravail,
)
from polytech.thermique.application.registre import JEUX_DE_MACROS
from polytech.thermique.infrastructure.latex import FichierDeMacros

_MATIERE: Final = Path("travaux") / "ThermiqueFondements"
_TRAVAIL: Final = _MATIERE / "tp-01-conductibilite"
_DEFAUT: Final = _TRAVAIL / "resultats.tex"


def cibles(arguments: list[str]) -> list[tuple[str | None, Path]]:
    """Les travaux a regenerer : `None` designe le TP 1, puis le registre."""
    if arguments:
        return [(None, Path(arguments[0]))]
    inscrits = [(t, _MATIERE / t / "resultats.tex") for t in sorted(JEUX_DE_MACROS)]
    return [(None, _DEFAUT), *inscrits]


def main(argv: list[str] | None = None) -> int:
    """Genere les macros ; retourne 0 en cas de succes."""
    arguments = sys.argv[1:] if argv is None else argv
    for travail, chemin in cibles(arguments):
        publication = FichierDeMacros(chemin=chemin)
        if travail is None:
            GenererMacros(publication=publication).executer()
            nombre = len(GenererMacros.macros_publiees())
        else:
            GenererMacrosDuTravail(publication=publication, travail=travail).executer()
            nombre = len(JEUX_DE_MACROS[travail]())
        sys.stdout.write(f"{nombre} macros dans {chemin}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
