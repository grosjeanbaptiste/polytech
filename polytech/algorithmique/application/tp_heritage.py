"""TP 6 : une hierarchie `Personne -> Etudiant, Prof -> Direction`.

Chaque classe derivee ajoute un champ a ceux de sa base, a la saisie
comme a l'affichage. Le doyen, membre de la direction, calcule la
moyenne des trois notes de chaque etudiant.
"""

from __future__ import annotations

from typing import Final

from polytech.algorithmique.application.releve import ligne, nombre
from polytech.algorithmique.domain.flux_d_entree import Clavier

ETUDIANT, PROF, SUPPRIMER, MOYENNES, CHERCHER, QUITTER = range(1, 7)
"""Les six points du menu, dans l'ordre de la feuille."""
NOTES: Final = 3
Fiche = tuple[str, ...]


def _lire(clavier: Clavier, champs: int) -> Fiche:
    return tuple(clavier.mot() for _ in range(champs))


def _etudiant(fiche: Fiche) -> str:
    """`Id/nom(n1,n2,n3)` : la base, puis ce que la derivee ajoute."""
    return f"{fiche[0]}/{fiche[1]}({','.join(fiche[2:])})"


def moyenne(fiche: Fiche) -> float:
    """La moyenne des trois notes d'un etudiant."""
    return sum(int(note) for note in fiche[2:]) / NOTES


def _rang(ensemble: list[Fiche], nom: str) -> int:
    rangs = [r for r, fiche in enumerate(ensemble) if fiche[1] == nom]
    return rangs[0] if rangs else -1


def derouler(saisie: str) -> tuple[str, ...]:
    """Les resultats que le programme affiche pour cette saisie, dans l'ordre."""
    clavier = Clavier(saisie)
    doyen = _lire(clavier, 4)
    etudiants: list[Fiche] = []
    profs: list[Fiche] = []
    lignes: list[str] = []

    def afficher(les_etudiants: bool) -> str:  # noqa: FBT001 -- deux listes, deux formes
        if les_etudiants:
            return ligne("ETUDIANTS", [_etudiant(fiche) for fiche in etudiants])
        return ligne("PROFS", ["/".join(fiche) for fiche in profs])

    while (choix := int(clavier.mot())) != QUITTER:
        if choix == ETUDIANT:
            etudiants.append(_lire(clavier, 2 + NOTES))
            lignes.append(afficher(True))  # noqa: FBT003 -- la liste des etudiants
        elif choix == PROF:
            profs.append(_lire(clavier, 3))
            lignes.append(afficher(False))  # noqa: FBT003 -- la liste des profs
        elif choix == SUPPRIMER:
            des_etudiants = clavier.mot() in {"E", "e"}
            ensemble, position = (
                (etudiants if des_etudiants else profs),
                int(clavier.mot()),
            )
            if 0 <= position < len(ensemble):
                del ensemble[position]
            else:
                lignes.append("Position invalide")
            lignes.append(afficher(des_etudiants))
        elif choix == MOYENNES:
            lignes.append(ligne("DOYEN", ["/".join(doyen)]))
            notes = [f"{fiche[1]}={nombre(moyenne(fiche))}" for fiche in etudiants]
            lignes.append(ligne("MOYENNES", notes))
        elif choix == CHERCHER:
            nom = clavier.mot()
            trouve = f"POSITION ETUDIANT = {_rang(etudiants, nom)}"
            lignes.append(f"{trouve} POSITION PROF = {_rang(profs, nom)}")
    return tuple(lignes)
