"""Point d'entree : regenere le fichier de macros consomme par le corrige.

Usage : ``python -m polytech.fluides [chemin]`` (defaut : ``resultats.tex``).
"""

from __future__ import annotations

import sys
from pathlib import Path

from polytech.fluides.application.generation import GenererMacros
from polytech.fluides.infrastructure.latex import FichierDeMacros

_DEFAUT = Path("resultats.tex")


def main(argv: list[str] | None = None) -> int:
    """Genere les macros ; retourne 0 en cas de succes."""
    arguments = sys.argv[1:] if argv is None else argv
    chemin = Path(arguments[0]) if arguments else _DEFAUT
    GenererMacros(publication=FichierDeMacros(chemin=chemin)).executer()
    sys.stdout.write(f"macros ecrites dans {chemin}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
