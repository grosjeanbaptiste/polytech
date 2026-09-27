"""Point d'entree : regenere tout ce que le corrige consomme.

Deux fichiers : les macros de resultats, et la table du champ de vitesse
que trace la figure des lignes de courant.

Usage : ``python -m polytech.fluides [chemin]``. Sans argument, la cible
est le travail en cours, range sous ``travaux/`` ; la table et l'apercu
prennent le meme repertoire.
"""

from __future__ import annotations

import sys
from pathlib import Path

from polytech.fluides.application.generation import GenererApercu, GenererMacros
from polytech.fluides.application.trace import GenererTableDuChamp
from polytech.fluides.infrastructure.apercu import ApercuDesMacros
from polytech.fluides.infrastructure.latex import FichierDeMacros
from polytech.fluides.infrastructure.table import FichierDeTable

_TRAVAIL = Path("travaux") / "MécaniqueFluides" / "activite-integration-1"
_DEFAUT = _TRAVAIL / "resultats.tex"
_TABLE = "champ_c.dat"
_APERCU = "resultats_apercu.tex"


def main(argv: list[str] | None = None) -> int:
    """Genere les macros ; retourne 0 en cas de succes."""
    arguments = sys.argv[1:] if argv is None else argv
    chemin = Path(arguments[0]) if arguments else _DEFAUT
    GenererMacros(publication=FichierDeMacros(chemin=chemin)).executer()
    table = chemin.parent / _TABLE
    GenererTableDuChamp(publication=FichierDeTable(chemin=table)).executer()
    apercu = chemin.parent / _APERCU
    GenererApercu(publication=ApercuDesMacros(chemin=apercu)).executer()
    sys.stdout.write(
        f"macros dans {chemin}, table dans {table}, apercu dans {apercu}\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
