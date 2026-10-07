"""TP 1 et TP 2 : ce que le programme a sept operations doit afficher.

Le modele lit la meme saisie que le programme C++ et applique les
operations du domaine : `TableauStatique` pour les suppressions et les
ajouts, `tri_a_bulles` pour le tri.
"""

from __future__ import annotations

from typing import Final

from polytech.algorithmique.application.releve import ligne, nombre
from polytech.algorithmique.domain.flux_d_entree import Clavier
from polytech.algorithmique.domain.tableaux import TableauStatique
from polytech.algorithmique.domain.tris import tri_a_bulles

NMAX: Final = 10
SAISIR, SUPPRIMER, AJOUTER, TRIER, EXTREMES, TRANSVASER, MILIEU, QUITTER = range(1, 9)
"""Les huit points du menu, dans l'ordre de la feuille."""


def _affiche(nom: str, tableau: TableauStatique) -> str:
    return ligne(nom, [nombre(v) for v in tableau.elements])


def _resume(valeurs: list[float]) -> tuple[float, float, float]:
    """Le minimum, le maximum et la moyenne : ce que le point 6 sort par reference."""
    return min(valeurs), max(valeurs), sum(valeurs) / len(valeurs)


def milieu_double_au_debut(tableau: TableauStatique) -> None:
    """Point 7 : retire l'element du milieu et met son double en tete."""
    retire = tableau.supprimer(tableau.taille // 2)
    tableau.inserer(0, 2 * retire)


def _operation_sur_tab1(
    choix: int, tab1: TableauStatique, clavier: Clavier
) -> list[str]:
    """Les points 2 a 5 ; chacun rend ce qu'il affiche."""
    if choix == SUPPRIMER:
        if tab1.taille:
            tab1.supprimer(tab1.taille - 1)
        if tab1.taille:
            tab1.supprimer(0)
    elif choix == AJOUTER:
        valeur = float(clavier.mot())
        if tab1.taille == tab1.capacite:
            return ["TAB1 est plein", _affiche("TAB1", tab1)]
        tab1.inserer(0, valeur)
    elif choix == TRIER:
        tab1.elements = list(tri_a_bulles(tab1.elements).valeurs)
    elif not tab1.taille:
        return ["TAB1 est vide"]
    else:
        mini, maxi, _ = _resume(tab1.elements)
        return [f"MIN = {nombre(mini)} MAX = {nombre(maxi)}"]
    return [_affiche("TAB1", tab1)]


def derouler(saisie: str) -> tuple[str, ...]:
    """Les resultats que le programme affiche pour cette saisie, dans l'ordre."""
    clavier = Clavier(saisie)
    n = int(clavier.mot())
    while not 1 <= n <= NMAX:
        n = int(clavier.mot())
    tab1, tab2 = TableauStatique(NMAX), TableauStatique(NMAX)
    lignes: list[str] = []
    resume: tuple[float, float, float] | None = None
    while (choix := int(clavier.mot())) != QUITTER:
        if choix == SAISIR:
            tab1.elements = [float(clavier.mot()) for _ in range(n)]
            lignes.append(_affiche("TAB1", tab1))
        elif choix in {SUPPRIMER, AJOUTER, TRIER, EXTREMES}:
            lignes += _operation_sur_tab1(choix, tab1, clavier)
        elif choix == TRANSVASER and tab1.taille:
            tab2.elements = list(tab1.elements)
            resume = _resume(tab2.elements)
            lignes.append(_affiche("TAB2", tab2))
        elif choix == TRANSVASER:
            lignes.append("TAB1 est vide")
        elif choix == MILIEU and tab2.taille:
            milieu_double_au_debut(tab2)
            lignes.append(_affiche("TAB2", tab2))
        elif choix == MILIEU:
            lignes.append("TAB2 est vide")
    if resume is not None:
        mini, maxi, moyenne = (nombre(v) for v in resume)
        fin = f"TAB2 au transvasement : MIN = {mini} MAX = {maxi} MOYENNE = {moyenne}"
        lignes.append(fin)
    return tuple(lignes)
