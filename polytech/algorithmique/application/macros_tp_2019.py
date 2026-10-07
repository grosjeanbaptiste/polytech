"""Macros du corrige des TP 1 a 6 (feuille de 2019). Prefixe `Tp`."""

from __future__ import annotations

from polytech.algorithmique.application.formatage import machine, session
from polytech.algorithmique.application.scenarios import SCENARIOS
from polytech.algorithmique.application.tp_objets import NMAX


def _lignes(nom: str) -> tuple[str, ...]:
    scenario = SCENARIOS[nom]
    return scenario.modele(scenario.saisie)


def _tableaux() -> dict[str, str]:
    lignes = _lignes("tp1_tp2")
    return {
        "TpTableauxMilieu": session(lignes[1:4]),
        "TpTableauxSuite": session(lignes[4:8]),
        "TpTableauxResume": machine(lignes[-1]),
    }


def _enregistrements() -> dict[str, str]:
    lignes = _lignes("tp3")
    return {
        "TpEnregistrementsTris": session(lignes[:5]),
        "TpEnregistrementsPositions": session(lignes[5:]),
    }


def _objets() -> dict[str, str]:
    lignes = _lignes("tp4")
    declaration = lignes.index(f"Declaration de PROF T_P[{NMAX}] :")
    fin = lignes.index("Fin du programme :")
    dynamique = _lignes("tp4_dynamique")
    return {
        "TpObjetsEtudiants": session(lignes[:declaration]),
        "TpObjetsConstruits": str(NMAX),
        "TpObjetsProfs": session(lignes[declaration + 1 + NMAX : fin]),
        "TpObjetsFin": session(lignes[fin + 1 :]),
        "TpObjetsDynamique": session(dynamique),
    }


def _conteneurs() -> dict[str, str]:
    lignes = _lignes("tp5")
    profs = next(r for r, texte in enumerate(lignes) if texte.startswith("PROFS"))
    heritage = _lignes("tp6")
    return {
        "TpConteneursEtudiants": session(lignes[:profs]),
        "TpConteneursProfs": session(lignes[profs:]),
        "TpHeritageSession": session(heritage),
        "TpMatieresSession": session(_lignes("tp6_matieres")),
    }


def macros_des_tp_de_2019() -> dict[str, str]:
    """Ce que chaque programme du corrige doit afficher sur son scenario."""
    return {**_tableaux(), **_enregistrements(), **_objets(), **_conteneurs()}
