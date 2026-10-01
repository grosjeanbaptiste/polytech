"""Point d'entree : regenere les macros que le corrige du TP 1 consomme.

Usage : ``python -m polytech.thermique [chemin]``. Sans argument, la cible
est le travail en cours, range sous ``travaux/``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from polytech.thermique.application.generation import GenererMacros
from polytech.thermique.infrastructure.latex import FichierDeMacros

_TRAVAIL: Final = Path("travaux") / "ThermiqueFondements" / "tp-01-conductibilite"
_DEFAUT: Final = _TRAVAIL / "resultats.tex"


def main(argv: list[str] | None = None) -> int:
    """Genere les macros ; retourne 0 en cas de succes."""
    arguments = sys.argv[1:] if argv is None else argv
    chemin = Path(arguments[0]) if arguments else _DEFAUT
    GenererMacros(publication=FichierDeMacros(chemin=chemin)).executer()
    nombre = len(GenererMacros.macros_publiees())
    sys.stdout.write(f"{nombre} macros dans {chemin}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
