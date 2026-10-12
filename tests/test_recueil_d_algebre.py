"""Le recueil d'algebre : dix chapitres d'exercices, et ceux qui restent.

`Mathematique/SYLLABUS/Syllabus_Algebre.pdf` est un cours de 188 pages dont
chaque chapitre se clot sur des exercices et une page de reponses finales.
L'inventaire des corriges (`outils.corriges`) ne compte que les enonces de
`EXS/`, `TP/` et `LABOS/` : ce recueil lui echappait, et la campagne des
corriges s'est dite close sans lui. Ce fichier tient le compte a part.

Le cliquet ne fait que descendre : un chapitre le quitte quand son corrige
pas a pas existe, est inscrit au registre des macros et cite le syllabus.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from polytech.maths.application.registre import JEUX_DE_MACROS

RACINE = Path(__file__).resolve().parent.parent
TRAVAUX = RACINE / "travaux" / "Mathematique"
RECUEIL = "Mathematique/SYLLABUS/Syllabus_Algebre.pdf"

Parties = tuple[str | None, ...]
CHAPITRES: dict[str, Parties] = {
    "I. Les nombres complexes": ("algebre-01-nombres-complexes",),
    "II. Les matrices": ("algebre-02-matrices",),
    "III. Les systemes lineaires": ("algebre-03-systemes-lineaires",),
    "IV. Les polynomes": ("algebre-04-polynomes",),
    "V. Les determinants": ("algebre-05-determinants",),
    "VI. Le rang d'une matrice": ("algebre-06-rang",),
    "VII. Les espaces vectoriels": ("algebre-07-espaces-vectoriels",),
    "VIII. Les applications lineaires": (
        "algebre-08a-applications-lineaires",
        "algebre-08b-matrices-associees",
        None,
    ),
    "IX. Les espaces metriques": (None,),
    "X. Les valeurs propres et les vecteurs propres": (None,),
}
"""Titre du chapitre, tel que le syllabus l'ecrit : ses dossiers de travail.

Le chapitre VIII a quarante-cinq problemes, dont treize sujets d'examen : il
est traite en trois dossiers, un par section d'exercices du syllabus --
VIII.4 (1 a 13), VIII.7 (14 a 32), VIII.8 (33 a 45).
"""

CHAPITRES_SANS_CORRIGE = 3
"""Le cliquet. Dix au depart, le 2026-10-12. Un chapitre le quitte quand
toutes ses parties ont leur dossier."""

PARTIES_SANS_CORRIGE = 3
"""Le cliquet des parties : celles du chapitre VIII se comptent une a une."""

CORRIGES = sorted(slug for parties in CHAPITRES.values() for slug in parties if slug)


def test_le_cliquet_des_chapitres_restants() -> None:
    restants = [titre for titre, parties in CHAPITRES.items() if None in parties]
    assert len(restants) == CHAPITRES_SANS_CORRIGE, restants


def test_le_cliquet_des_parties_restantes() -> None:
    manquantes = sum(parties.count(None) for parties in CHAPITRES.values())
    assert manquantes == PARTIES_SANS_CORRIGE


@pytest.mark.parametrize("slug", CORRIGES)
def test_un_chapitre_corrige_a_son_dossier_et_ses_macros(slug: str) -> None:
    activite = tomllib.loads((TRAVAUX / slug / "activite.toml").read_text("utf-8"))
    assert activite["enonce"] == RECUEIL
    assert activite["traitement"] == "entier"
    assert slug in JEUX_DE_MACROS


def test_aucun_dossier_du_recueil_n_est_oublie_de_la_table() -> None:
    presents = sorted(d.name for d in TRAVAUX.glob("algebre-*") if d.is_dir())
    assert presents == CORRIGES
