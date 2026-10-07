"""TP 4, feuille de 2017 : ce que le fichier de demarrage complete doit afficher.

Le fichier fourni gere les etudiants, avec deux fonctions laissees
vides. Sa partie II, sur les profs, est a ecrire : un constructeur sans
argument pour les cases du tableau, qui ne numerote pas, et un second
qui cree un vrai prof et lui donne son Id. Le modele suit les cases une
a une, copies comprises.
"""

from __future__ import annotations

from typing import Final

from polytech.algorithmique.domain.enregistrements import trier_selon
from polytech.algorithmique.domain.flux_d_entree import Clavier

NMAX: Final = 50
AJOUTER, SUPPRIMER, CHERCHER, TRIER, QUITTER = range(1, 6)
Case = tuple[int, str]
VIDE: Final[Case] = (0, "")
"""Une case que le constructeur sans argument a batie : pas de numero."""


def _inserer(cases: list[Case], taille: int, position: int, nouveau: Case) -> None:
    """Decale vers la droite a partir de la fin, puis ecrit a la position."""
    for rang in range(taille, position, -1):
        cases[rang] = cases[rang - 1]
    cases[position] = nouveau


def _supprimer_le_debut(cases: list[Case], taille: int) -> None:
    for rang in range(taille - 1):
        cases[rang] = cases[rang + 1]


def _rang(cases: list[Case], taille: int, identifiant: int) -> int:
    rangs = [r for r, (i, _) in enumerate(cases[:taille]) if i == identifiant]
    return rangs[0] if rangs else -1


def _etudiants(clavier: Clavier) -> list[str]:
    """La partie I : le menu du fichier fourni, ses deux fonctions completees."""
    cases, taille, lignes = [VIDE] * NMAX, 0, []
    while (choix := int(clavier.mot())) != QUITTER:
        if choix == AJOUTER:
            ou = clavier.mot().upper()
            if ou in {"D", "F"}:
                nouveau = (int(clavier.mot()), clavier.mot())
                _inserer(cases, taille, 0 if ou == "D" else taille, nouveau)
                taille += 1
            else:
                lignes.append("Erreur ....")
        elif choix == SUPPRIMER and taille:
            _supprimer_le_debut(cases, taille)
            taille -= 1
        elif choix == CHERCHER:
            lignes.append(f"POSITION = {_rang(cases, taille, int(clavier.mot()))}")
            continue
        elif choix == TRIER and taille:
            cases[:taille] = trier_selon(cases[:taille], lambda case: case[1])
        elif choix == TRIER:
            lignes.append("C est vide ...!!")
            continue
        else:
            continue
        lignes += [f"Code:{i} nom:{nom}" for i, nom in cases[:taille]]
    return lignes


def _trier_les_profs(cases: list[Case], taille: int) -> list[str]:
    """Le tri a bulles ; chaque echange detruit le temporaire qui l'a servi."""
    detruits, echange = [], True
    while echange:
        echange = False
        for rang in range(taille - 1):
            if cases[rang][1] > cases[rang + 1][1]:
                detruits.append(f"- PROF {cases[rang][0]} detruit")
                cases[rang], cases[rang + 1] = cases[rang + 1], cases[rang]
                echange = True
    return detruits


def _profs(clavier: Clavier) -> list[str]:
    """La partie II : les memes operations, sur des objets numerotes."""
    cases, taille, numerotes, lignes = [VIDE] * NMAX, 0, 0, []
    while (choix := int(clavier.mot())) != QUITTER:
        if choix == AJOUTER and taille < NMAX:
            au_milieu = clavier.mot().upper() == "M"
            numerotes += 1
            nouveau = (numerotes, clavier.mot())
            _inserer(cases, taille, taille // 2 if au_milieu else 0, nouveau)
            taille += 1
            lignes.append(f"- PROF {numerotes} detruit")
        elif choix == SUPPRIMER and taille:
            _supprimer_le_debut(cases, taille)
            taille -= 1
        elif choix == CHERCHER:
            lignes.append(f"POSITION = {_rang(cases, taille, int(clavier.mot()))}")
            continue
        elif choix == TRIER:
            lignes += _trier_les_profs(cases, taille)
        else:
            continue
        lignes += [f"Id:{i} nom:{nom}" for i, nom in cases[:taille]]
    lignes.append("Fin du programme :")
    return lignes + [f"- PROF {i} detruit" for i, _ in reversed(cases) if i]


def derouler(saisie: str) -> tuple[str, ...]:
    """Les resultats que le programme affiche pour cette saisie, dans l'ordre."""
    clavier = Clavier(saisie)
    return (*_etudiants(clavier), *_profs(clavier))
