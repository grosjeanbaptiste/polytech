"""TP 5 : les etudiants dans une `list`, les profs dans un `vector`.

Le meme traitement du menu sert aux deux conteneurs ; seuls changent ce
qu'on saisit, ce qu'on affiche, et ce que `operator==` compare : l'Id et
le nom pour un etudiant, le nom et le service pour un prof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from polytech.algorithmique.application.releve import ligne
from polytech.algorithmique.domain.flux_d_entree import Clavier

if TYPE_CHECKING:
    from collections.abc import Callable

CHOISIR, SAISIR, AJOUTER, CHERCHER, PAR_NOM, PAR_ID, QUITTER = range(7)
"""Les sept points du menu, de 0 a 6."""
Fiche = tuple[str, ...]
"""Les champs saisis : `(Id, nom)` ou `(Id, nom, service)`."""


@dataclass(frozen=True, slots=True)
class Sorte:
    """Ce qui distingue les deux conteneurs."""

    etiquette: str
    champs: int
    identite: Callable[[Fiche], Fiche]
    """Ce que `operator==` compare."""


ETUDIANTS: Final = Sorte("ETUDIANTS", 2, lambda fiche: fiche)
PROFS: Final = Sorte("PROFS", 3, lambda fiche: fiche[1:])


def _lire(clavier: Clavier, sorte: Sorte) -> Fiche:
    return tuple(clavier.mot() for _ in range(sorte.champs))


def _traiter(
    choix: int, ensemble: list[Fiche], sorte: Sorte, clavier: Clavier
) -> list[str]:
    """Un point du menu sur un conteneur ; rend ce qu'il affiche."""
    if choix == SAISIR:
        ensemble[:] = [_lire(clavier, sorte) for _ in range(int(clavier.mot()))]
    elif choix == AJOUTER:
        au_debut = clavier.mot() in {"D", "d"}
        ensemble.insert(0 if au_debut else len(ensemble), _lire(clavier, sorte))
    elif choix == CHERCHER:
        cherche = sorte.identite(_lire(clavier, sorte))
        present = any(sorte.identite(fiche) == cherche for fiche in ensemble)
        return ["PRESENT" if present else "ABSENT"]
    elif choix == PAR_NOM:
        ensemble.sort(key=lambda fiche: fiche[1])
    elif choix == PAR_ID:
        ensemble.sort(key=lambda fiche: int(fiche[0]))
    else:
        return []
    return [ligne(sorte.etiquette, ["/".join(fiche) for fiche in ensemble])]


def derouler(saisie: str) -> tuple[str, ...]:
    """Les resultats que le programme affiche pour cette saisie, dans l'ordre."""
    clavier = Clavier(saisie)
    ensembles: dict[str, list[Fiche]] = {"ETUDIANTS": [], "PROFS": []}
    sorte = ETUDIANTS
    lignes: list[str] = []
    while (choix := int(clavier.mot())) != QUITTER:
        if choix == CHOISIR:
            sorte = PROFS if clavier.mot() in {"P", "p"} else ETUDIANTS
        else:
            lignes += _traiter(choix, ensembles[sorte.etiquette], sorte, clavier)
    return tuple(lignes)
