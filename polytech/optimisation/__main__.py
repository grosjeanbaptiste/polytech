"""Point d'entree : regenere les macros que les corriges consomment.

Usage : ``python -m polytech.optimisation [travail...]``. Sans
argument, tous les travaux du registre sont regeneres -- ce qui est le
geste utile apres une modification du domaine, puisqu'elle peut toucher
plusieurs exercices a la fois.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from polytech.optimisation.application.generation import GenererMacros
from polytech.optimisation.application.registre import JEUX_DE_MACROS
from polytech.optimisation.infrastructure.latex import FichierDeMacros

TRAVAUX: Final = Path("travaux") / "Optimisation"


def main(argv: list[str] | None = None) -> int:
    """Genere les macros des travaux demandes ; rend 0 en cas de succes."""
    arguments = sys.argv[1:] if argv is None else argv
    travaux = arguments or sorted(JEUX_DE_MACROS)
    for travail in travaux:
        chemin = TRAVAUX / travail / "resultats.tex"
        commande = GenererMacros(
            publication=FichierDeMacros(chemin=chemin),
            travail=travail,
        )
        commande.executer()
        sys.stdout.write(f"{travail} : macros ecrites dans {chemin}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
