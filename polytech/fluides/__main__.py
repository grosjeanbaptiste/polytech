"""Point d'entree : regenere tout ce que le corrige consomme.

Deux fichiers : les macros de resultats, et la table du champ de vitesse
que trace la figure des lignes de courant.

Usage : ``python -m polytech.fluides [chemin]`` (defaut : ``resultats.tex`` ;
la table prend le meme repertoire, sous le nom ``champ_c.dat``).
"""

from __future__ import annotations

import sys
from pathlib import Path

from polytech.fluides.application.generation import GenererMacros
from polytech.fluides.application.trace import GenererTableDuChamp
from polytech.fluides.infrastructure.latex import FichierDeMacros
from polytech.fluides.infrastructure.table import FichierDeTable

_DEFAUT = Path("resultats.tex")
_TABLE = "champ_c.dat"


def main(argv: list[str] | None = None) -> int:
    """Genere les macros ; retourne 0 en cas de succes."""
    arguments = sys.argv[1:] if argv is None else argv
    chemin = Path(arguments[0]) if arguments else _DEFAUT
    GenererMacros(publication=FichierDeMacros(chemin=chemin)).executer()
    table = chemin.parent / _TABLE
    GenererTableDuChamp(publication=FichierDeTable(chemin=table)).executer()
    sys.stdout.write(f"macros ecrites dans {chemin}, table dans {table}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
