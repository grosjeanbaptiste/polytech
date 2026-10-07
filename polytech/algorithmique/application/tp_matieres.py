"""TP 6, prolongement : des matieres au choix, et un prof par matiere.

Chaque etudiant choisit le nombre de ses matieres et leur titre ; un
prof n'en enseigne qu'une. « Les etudiants qui ont cours avec le prof
X » sont ceux qui suivent sa matiere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.algorithmique.application.releve import ligne, nombre
from polytech.algorithmique.domain.flux_d_entree import Clavier

if TYPE_CHECKING:
    from collections.abc import Mapping

ETUDIANT, PROF, MOYENNES, COURS, QUITTER = range(1, 6)
"""Les cinq points du menu."""
Notes = dict[str, int]
SANS_MATIERE: Final = 0.0
"""La moyenne d'un etudiant qui n'a choisi aucune matiere."""


def moyenne(notes: Mapping[str, int]) -> float:
    """La moyenne des notes, quel que soit leur nombre."""
    return sum(notes.values()) / len(notes) if notes else SANS_MATIERE


def etudiants_de(
    etudiants: Mapping[str, Mapping[str, int]], profs: Mapping[str, str], prof: str
) -> tuple[str, ...]:
    """Les noms des etudiants qui suivent la matiere de ce prof."""
    if prof not in profs:
        return ()
    return tuple(nom for nom, notes in etudiants.items() if profs[prof] in notes)


def derouler(saisie: str) -> tuple[str, ...]:
    """Les resultats que le programme affiche pour cette saisie, dans l'ordre."""
    clavier = Clavier(saisie)
    etudiants: dict[str, Notes] = {}
    profs: dict[str, str] = {}
    lignes: list[str] = []
    while (choix := int(clavier.mot())) != QUITTER:
        if choix == ETUDIANT:
            nom, combien = clavier.mot(), int(clavier.mot())
            etudiants[nom] = {clavier.mot(): int(clavier.mot()) for _ in range(combien)}
        elif choix == PROF:
            nom = clavier.mot()  # le nom d'abord : une affectation lit sa droite
            profs[nom] = clavier.mot()
        elif choix == MOYENNES:
            notes = [f"{nom}={nombre(moyenne(n))}" for nom, n in etudiants.items()]
            lignes.append(ligne("MOYENNES", notes))
        elif choix == COURS:
            lignes.append(
                ligne("ETUDIANTS", etudiants_de(etudiants, profs, clavier.mot()))
            )
    return tuple(lignes)
