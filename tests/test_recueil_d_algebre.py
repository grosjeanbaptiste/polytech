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

CHAPITRES: dict[str, str | None] = {
    "I. Les nombres complexes": "algebre-01-nombres-complexes",
    "II. Les matrices": "algebre-02-matrices",
    "III. Les systemes lineaires": None,
    "IV. Les polynomes": None,
    "V. Les determinants": None,
    "VI. Le rang d'une matrice": None,
    "VII. Les espaces vectoriels": None,
    "VIII. Les applications lineaires": None,
    "IX. Les espaces metriques": None,
    "X. Les valeurs propres et les vecteurs propres": None,
}
"""Titre du chapitre, tel que le syllabus l'ecrit : son dossier de travail."""

CHAPITRES_SANS_CORRIGE = 8
"""Le cliquet. Dix au depart, le 2026-10-12."""

CORRIGES = sorted(slug for slug in CHAPITRES.values() if slug is not None)


def test_le_cliquet_des_chapitres_restants() -> None:
    restants = [titre for titre, slug in CHAPITRES.items() if slug is None]
    assert len(restants) == CHAPITRES_SANS_CORRIGE, restants


@pytest.mark.parametrize("slug", CORRIGES)
def test_un_chapitre_corrige_a_son_dossier_et_ses_macros(slug: str) -> None:
    activite = tomllib.loads((TRAVAUX / slug / "activite.toml").read_text("utf-8"))
    assert activite["enonce"] == RECUEIL
    assert activite["traitement"] == "entier"
    assert slug in JEUX_DE_MACROS


def test_aucun_dossier_du_recueil_n_est_oublie_de_la_table() -> None:
    presents = sorted(d.name for d in TRAVAUX.glob("algebre-*") if d.is_dir())
    assert presents == CORRIGES
