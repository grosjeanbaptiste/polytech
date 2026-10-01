"""Ce qu'un corrige C++ des TP devra retrouver, calcule independamment.

Moyenne generale, classement, effectif par section, sections enseignees :
les quatre resultats que les TP 1 a 6 demandent de produire. Les calculer
ici donne au corrige quelque chose a **prouver** -- sans oracle, un
corrige ne peut qu'affirmer sa propre sortie, et une faute de tri ou de
moyenne y passe inapercue.

Un controle intrinseque les lie : la somme des effectifs par section doit
valoir l'effectif total. Aucun etudiant ne se perd au filtrage, et c'est
exactement la faute qu'un `if` mal ecrit produit.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polytech.algorithmique.domain.dossier import Etudiant, Professeur


def _exiger_des_etudiants(etudiants: tuple[Etudiant, ...]) -> None:
    if not etudiants:
        msg = "aucun etudiant : le calcul n'a pas d'objet"
        raise ValueError(msg)


def moyenne_generale(etudiants: tuple[Etudiant, ...]) -> float:
    """La moyenne des moyennes individuelles.

    Et non la moyenne de toutes les notes : les deux coincident seulement
    si chaque etudiant a le meme nombre de notes. L'enonce le garantit par
    son entete, mais s'appuyer dessus sans le dire serait fragile.
    """
    _exiger_des_etudiants(etudiants)
    return sum(etudiant.moyenne() for etudiant in etudiants) / len(etudiants)


def classement_par_moyenne(etudiants: tuple[Etudiant, ...]) -> tuple[Etudiant, ...]:
    """Les etudiants par moyenne decroissante, a nom egal par ordre alphabetique.

    Le second critere n'est pas decoratif : sans lui, deux etudiants de
    meme moyenne se classeraient dans l'ordre du fichier, et deux corriges
    corrects rendraient des sorties differentes.
    """
    _exiger_des_etudiants(etudiants)
    return tuple(sorted(etudiants, key=lambda e: (-e.moyenne(), e.nom)))


def effectif_par_section(etudiants: tuple[Etudiant, ...]) -> dict[str, int]:
    """Le nombre d'etudiants par section.

    Leur somme doit valoir l'effectif total : c'est le controle intrinseque
    du filtrage, et la faute qu'un `if` mal ecrit produit.
    """
    _exiger_des_etudiants(etudiants)
    return dict(Counter(etudiant.section for etudiant in etudiants))


def sections_enseignees(professeurs: tuple[Professeur, ...]) -> tuple[str, ...]:
    """Les sections couvertes par le corps professoral, sans doublon et triees.

    Deux professeurs peuvent partager une section : les compter deux fois
    ferait croire a une couverture plus large qu'elle n'est.
    """
    if not professeurs:
        msg = "aucun professeur : aucune section n'est enseignee"
        raise ValueError(msg)
    return tuple(sorted({s for p in professeurs for s in p.sections}))
