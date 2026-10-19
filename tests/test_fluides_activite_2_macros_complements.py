"""Les macros que le corrige enrichi de l'activite 2 consomme en plus.

Les valeurs attendues sont refaites a la main dans ce fichier, a partir des
donnees du cours que `tests/test_fluides_activite_2_hypotheses.py` cite.
"""

from __future__ import annotations

import pytest

from polytech.fluides.application.macros_activite_2_complements import (
    macros_complements,
)
from polytech.fluides.application.registre import macros_du_travail


@pytest.fixture(scope="module")
def m() -> dict[str, str]:
    return macros_complements()


def test_les_proprietes_du_cours(m: dict[str, str]) -> None:
    assert m["FilmMasseVolumiqueHuile"] == "900"
    assert m["FilmViscositeHuileMilli"] == "100"
    assert m["FilmViscositeAir"] == "1.83e-5"
    assert m["FilmRapportViscosites"] == "1.8e-4"


def test_la_poussee_de_l_air(m: dict[str, str]) -> None:
    assert m["FilmMasseVolumiqueAir"] == "1.21"
    assert m["FilmCorrectionPousseePourcent"] == "0.13"


def test_le_reynolds_du_film(m: dict[str, str]) -> None:
    assert m["FilmReynoldsMillimetre"] == "0.26"
    assert m["FilmEpaisseurReynoldsUnite"] == "1.56"


def test_le_reservoir(m: dict[str, str]) -> None:
    assert m["ReservoirErreurDixiemePourcent"] == "0.01"
    assert m["ReservoirErreurTiersPourcent"] == "1.23"


def test_le_bateau(m: dict[str, str]) -> None:
    assert m["BateauElasticiteDiametre"] == "-3.8"
    assert m["BateauElasticiteDebit"] == "2.9"
    assert m["BateauEcartBuseEtroitePourcent"] == "8"
    assert m["BateauEcartBuseLargePourcent"] == "-7.2"


def test_les_recoupements_sont_en_notation_scientifique(m: dict[str, str]) -> None:
    for cle in ("RecoupementProfil", "RecoupementRemplissage", "RecoupementPuissance"):
        assert "e-" in m[cle]
    assert float(m["RecoupementDebit"]) == 0.0


def test_le_registre_publie_les_complements_avec_le_reste() -> None:
    jeu = macros_du_travail("activite-integration-2")
    assert "FilmReynoldsMillimetre" in jeu
    assert "BateauVitesseEjection" in jeu
