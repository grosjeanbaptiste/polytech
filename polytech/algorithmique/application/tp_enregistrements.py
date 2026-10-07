"""TP 3 : ce que le programme sur un tableau d'enregistrements doit afficher.

Les operations viennent du domaine : `trier_selon`, `sans_doublons`,
`position_de`, et `TableauStatique` pour effacer et ajouter a une
position. « Permuter deux a deux » suit la feuille de TP : le premier
avec le dernier, le deuxieme avec l'avant-dernier.
"""

from __future__ import annotations

from typing import Final

from polytech.algorithmique.application.releve import ligne
from polytech.algorithmique.domain import algorithmes_stl
from polytech.algorithmique.domain.enregistrements import (
    Article,
    position_de,
    sans_doublons,
    trier_selon,
)
from polytech.algorithmique.domain.flux_d_entree import Clavier
from polytech.algorithmique.domain.tableaux import TableauStatique, indice_valide

NMAX: Final = 10
SAISIR, PERMUTER, PAR_CODE, PAR_NOM, EFFACER, AJOUTER, CHERCHER, DOUBLONS, QUITTER = (
    range(1, 10)
)
"""Les neuf points du menu, dans l'ordre de la feuille."""
ABSENT: Final = -1
"""Ce que le programme affiche quand l'element cherche n'existe pas."""


def _article(clavier: Clavier) -> Article:
    return Article(clavier.mot(), clavier.mot())


def _affiche(tableau: TableauStatique) -> str:
    return ligne("TAB", [f"{a.nom}/{a.code}" for a in tableau.elements])


def _saisir(clavier: Clavier) -> list[Article]:
    """La taille est redemandee tant qu'elle n'est pas entre 0 et dix."""
    n = int(clavier.mot())
    while not 0 <= n <= NMAX:
        n = int(clavier.mot())
    return [_article(clavier) for _ in range(n)]


def _a_une_position(
    choix: int, tableau: TableauStatique, clavier: Clavier
) -> list[str]:
    """Points 5 et 6 : effacer ou ajouter a la position que l'utilisateur donne."""
    position = int(clavier.mot())
    if choix == EFFACER:
        if not indice_valide(position, tableau.taille):
            return ["Position invalide"]
        tableau.supprimer(position)
        return []
    nouveau = _article(clavier)
    plein = tableau.taille == tableau.capacite
    if plein or not indice_valide(position, tableau.taille + 1):
        return ["Ajout impossible"]
    tableau.inserer(position, nouveau)
    return []


def derouler(saisie: str) -> tuple[str, ...]:
    """Les resultats que le programme affiche pour cette saisie, dans l'ordre."""
    clavier = Clavier(saisie)
    tableau = TableauStatique(NMAX)
    lignes: list[str] = []
    while (choix := int(clavier.mot())) != QUITTER:
        if choix == SAISIR:
            tableau.elements = _saisir(clavier)
        elif choix == PERMUTER:
            algorithmes_stl.reverse(tableau.elements)
        elif choix in {PAR_CODE, PAR_NOM}:
            cle = (lambda a: a.code) if choix == PAR_CODE else (lambda a: a.nom)
            tableau.elements = list(trier_selon(tableau.elements, cle))
        elif choix in {EFFACER, AJOUTER}:
            lignes += _a_une_position(choix, tableau, clavier)
        elif choix == CHERCHER:
            cherche = _article(clavier)
            rang = position_de(tableau.elements, cherche.nom, cherche.code)
            trouve = rang if rang < tableau.taille else ABSENT
            lignes.append(f"POSITION = {trouve}")
            continue
        elif choix == DOUBLONS:
            tableau.elements = list(sans_doublons(tableau.elements))
        else:
            continue
        lignes.append(_affiche(tableau))
    return tuple(lignes)
