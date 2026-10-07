"""Macros du corrige de la feuille de 2017 (TP 4 a 6). Prefixe `TpAncien`."""

from __future__ import annotations

from polytech.algorithmique.application.formatage import session
from polytech.algorithmique.application.scenarios import SCENARIOS
from polytech.algorithmique.application.tp_demarrage import NMAX


def macros_des_tp_de_2017() -> dict[str, str]:
    """Ce que le fichier de demarrage complete doit afficher sur son scenario."""
    scenario = SCENARIOS["tp4_demarrage"]
    lignes = scenario.modele(scenario.saisie)
    profs = next(r for r, texte in enumerate(lignes) if texte.startswith("- PROF"))
    fin = lignes.index("Fin du programme :")
    return {
        "TpAncienCases": str(NMAX),
        "TpAncienEtudiants": session(lignes[:profs]),
        "TpAncienProfs": session(lignes[profs:fin]),
        "TpAncienFin": session(lignes[fin + 1 :]),
    }
