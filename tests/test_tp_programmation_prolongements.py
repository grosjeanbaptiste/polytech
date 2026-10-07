"""Programmation et algorithmique : les « s'il vous reste du temps » des TP 4 et 6.

Source : `ProgrammationAlgorithmique/TP/TP_01_a_06_2019.pdf`, pages 10 et
12. Deux questions ouvertes y recoivent un programme : le tableau de
profs defini dynamiquement (`test_tp_programmation_modeles.py`), et les
matieres au choix, traitees ici.
"""

from __future__ import annotations

from polytech.algorithmique.application import tp_matieres
from polytech.algorithmique.application.scenarios import SCENARIOS


def test_tp6_chaque_etudiant_a_ses_matieres() -> None:
    """Zoe en suit deux, Ali quatre, Bob aucune : la moyenne suit le nombre."""
    lignes = tp_matieres.derouler(SCENARIOS["tp6_matieres"].saisie)
    assert lignes[0] == "MOYENNES : zoe=15 ali=11.25 bob=0"


def test_tp6_les_etudiants_qui_ont_cours_avec_un_prof() -> None:
    """Turing enseigne l'info, Gauss les maths ; Curie n'est pas un prof connu."""
    lignes = tp_matieres.derouler(SCENARIOS["tp6_matieres"].saisie)
    assert lignes[1:] == ("ETUDIANTS : zoe ali", "ETUDIANTS : zoe", "ETUDIANTS :")


def test_tp6_les_etudiants_d_un_prof_par_le_domaine() -> None:
    etudiants = {"zoe": {"math": 14, "info": 16}, "ali": {"info": 12}}
    assert tp_matieres.etudiants_de(etudiants, {"turing": "info"}, "turing") == (
        "zoe",
        "ali",
    )
    assert tp_matieres.etudiants_de(etudiants, {"turing": "info"}, "gauss") == ()
