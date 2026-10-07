"""TP 4 : un tableau d'objets, et le moment ou chacun est construit puis detruit.

`PROF T_P[10]` construit dix objets d'un coup, avant toute saisie : le
constructeur qui numerote les a deja servis quand le premier prof est
ajoute. Une affectation recopie l'Id ; une case liberee par un decalage
garde donc sa copie. Le modele suit ces deux faits case par case.
"""

from __future__ import annotations

from typing import Final

from polytech.algorithmique.application.releve import ligne
from polytech.algorithmique.domain.enregistrements import trier_selon
from polytech.algorithmique.domain.flux_d_entree import Clavier

NMAX: Final = 10
AJOUTER, SUPPRIMER, CHERCHER, TRIER, QUITTER = range(1, 6)
"""Les points du menu commun aux etudiants et aux profs."""
Case = tuple[int, str]
"""Un Id et un nom : ce que porte une case du tableau."""


def _affiche(nom: str, cases: list[Case], taille: int) -> str:
    return ligne(nom, [f"{identifiant}/{qui}" for identifiant, qui in cases[:taille]])


def _decaler_a_droite(cases: list[Case], taille: int, nouveau: Case) -> None:
    """`tab[i] = tab[i-1]` en partant de la fin, puis la case 0."""
    for rang in range(taille, 0, -1):
        cases[rang] = cases[rang - 1]
    cases[0] = nouveau


def _decaler_a_gauche(cases: list[Case], taille: int) -> None:
    """`tab[i] = tab[i+1]` : la derniere case occupee garde sa copie."""
    for rang in range(taille - 1):
        cases[rang] = cases[rang + 1]


def _gerer(
    clavier: Clavier, cases: list[Case], nom: str, crees: int | None
) -> list[str]:
    """La boucle du menu ; `crees` compte les PROF construits, `None` sinon."""
    lignes: list[str] = []
    taille = 0
    while (choix := int(clavier.mot())) != QUITTER:
        if choix == AJOUTER:
            if crees is None:
                nouveau = (int(clavier.mot()), clavier.mot())
            else:
                crees += 1
                lignes.append(f"+ PROF {crees} construit")
                nouveau = (crees, clavier.mot())
            if taille < NMAX:
                _decaler_a_droite(cases, taille, nouveau)
                taille += 1
            else:
                lignes.append("Tableau plein")
            if crees is not None:
                lignes.append(f"- PROF {crees} detruit")
        elif choix == SUPPRIMER and taille:
            _decaler_a_gauche(cases, taille)
            taille -= 1
        elif choix == SUPPRIMER:
            lignes.append("Tableau vide")
        elif choix == CHERCHER:
            cherche = int(clavier.mot())
            rangs = [r for r, (i, _) in enumerate(cases[:taille]) if i == cherche]
            lignes.append(f"POSITION = {rangs[0] if rangs else -1}")
            continue
        elif choix == TRIER:
            cases[:taille] = trier_selon(cases[:taille], lambda case: case[1])
        lignes.append(_affiche(nom, cases, taille))
    return lignes


def derouler(saisie: str) -> tuple[str, ...]:
    """Les resultats du programme a tableaux statiques, pour cette saisie."""
    clavier = Clavier(saisie)
    lignes = _gerer(clavier, [(0, "")] * NMAX, "T_E", None)
    lignes.append(f"Declaration de PROF T_P[{NMAX}] :")
    profs: list[Case] = [(rang, "") for rang in range(1, NMAX + 1)]
    lignes += [f"+ PROF {identifiant} construit" for identifiant, _ in profs]
    lignes += _gerer(clavier, profs, "T_P", NMAX)
    lignes.append("Fin du programme :")
    lignes += [f"- PROF {identifiant} detruit" for identifiant, _ in reversed(profs)]
    return tuple(lignes)


def derouler_dynamique(saisie: str) -> tuple[str, ...]:
    """Les resultats du programme a tableau de pointeurs : `new` et `delete`."""
    clavier = Clavier(saisie)
    profs: list[Case] = []
    crees = 0
    lignes = [f"Declaration de PROF* T_P[{NMAX}] : rien n'est construit"]
    while (choix := int(clavier.mot())) != QUITTER - 1:
        if choix == AJOUTER and len(profs) < NMAX:
            crees += 1
            lignes.append(f"+ PROF {crees} construit")
            profs.insert(0, (crees, clavier.mot()))
        elif choix == AJOUTER:
            lignes.append("Tableau plein")
        elif choix == SUPPRIMER and profs:
            lignes.append(f"- PROF {profs.pop(0)[0]} detruit")
        elif choix == SUPPRIMER:
            lignes.append("Tableau vide")
        elif choix == CHERCHER:
            cherche = (int(clavier.mot()), clavier.mot())
            lignes.append(
                f"POSITION = {profs.index(cherche) if cherche in profs else -1}"
            )
            continue
        lignes.append(_affiche("T_P", profs, len(profs)))
    lignes.append("Fin du programme :")
    lignes += [f"- PROF {identifiant} detruit" for identifiant, _ in profs]
    return tuple(lignes)
