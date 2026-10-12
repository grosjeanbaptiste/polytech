"""Point d'entree : regenere tout ce que le corrige consomme.

Deux fichiers : les macros de resultats, et la table du champ de vitesse
que trace la figure des lignes de courant.

Usage : ``python -m polytech.fluides [chemin]``. Sans argument, la cible
est l'activite 1, rangee sous ``travaux/`` -- la table et l'apercu
prennent le meme repertoire --, puis chaque travail inscrit au registre.
Avec un chemin, seule l'activite 1 est regeneree, vers ce chemin.
"""

from __future__ import annotations

import sys
from pathlib import Path

from polytech.fluides.application.generation import GenererApercu, GenererMacros
from polytech.fluides.application.generation_du_travail import GenererMacrosDuTravail
from polytech.fluides.application.registre import JEUX_DE_MACROS
from polytech.fluides.application.trace import GenererTableDuChamp
from polytech.fluides.infrastructure.apercu import ApercuDesMacros
from polytech.fluides.infrastructure.latex import FichierDeMacros
from polytech.fluides.infrastructure.table import FichierDeTable

_MATIERE = Path("travaux") / "MecaniqueFluides"
_TRAVAIL = _MATIERE / "activite-integration-1"
_DEFAUT = _TRAVAIL / "resultats.tex"
_TABLE = "champ_c.dat"
_APERCU = "resultats_apercu.tex"


def cibles(arguments: list[str]) -> list[tuple[str | None, Path]]:
    """Les travaux a regenerer : `None` designe l'activite 1, puis le registre."""
    if arguments:
        return [(None, Path(arguments[0]))]
    inscrits = [(t, _MATIERE / t / "resultats.tex") for t in sorted(JEUX_DE_MACROS)]
    return [(None, _DEFAUT), *inscrits]


def _activite_un(chemin: Path) -> None:
    """Les macros de l'activite 1, sa table et son apercu."""
    GenererMacros(publication=FichierDeMacros(chemin=chemin)).executer()
    table = chemin.parent / _TABLE
    GenererTableDuChamp(publication=FichierDeTable(chemin=table)).executer()
    apercu = chemin.parent / _APERCU
    GenererApercu(publication=ApercuDesMacros(chemin=apercu)).executer()
    sys.stdout.write(
        f"macros dans {chemin}, table dans {table}, apercu dans {apercu}\n",
    )


def main(argv: list[str] | None = None) -> int:
    """Genere les macros ; retourne 0 en cas de succes."""
    arguments = sys.argv[1:] if argv is None else argv
    for travail, chemin in cibles(arguments):
        if travail is None:
            _activite_un(chemin)
            continue
        publication = FichierDeMacros(chemin=chemin)
        GenererMacrosDuTravail(publication=publication, travail=travail).executer()
        sys.stdout.write(f"macros dans {chemin}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
