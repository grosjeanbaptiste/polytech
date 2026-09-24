"""Les donnees de l'enonce ne sont transcrites qu'une fois, dans enonce.py."""

from collections.abc import Mapping

import pytest

from fluides.application.generation import GenererMacros


class _EnMemoire:
    def __init__(self) -> None:
        self.macros: dict[str, str] = {}

    def publier(self, macros: Mapping[str, str]) -> None:
        self.macros = dict(macros)


def _macros() -> dict[str, str]:
    publication = _EnMemoire()
    GenererMacros(publication=publication).executer()
    return publication.macros


MACROS = _macros()


@pytest.mark.parametrize(
    ("macro", "attendu"),
    [
        ("MasseVolumiqueEau", "1000"),
        ("MasseVolumiqueHuile", "850"),
        ("MasseVolumiqueMercure", "13600"),
        ("HauteurUn", "0.10"),
        ("HauteurDeux", "0.20"),
        ("HauteurTrois", "0.35"),
        ("Pesanteur", "9.81"),
        ("DensiteHuile", "0.85"),
        ("DensiteMercure", "13.6"),
    ],
)
def test_les_donnees_sont_publiees(macro: str, attendu: str) -> None:
    assert MACROS[macro] == attendu


def test_les_champs_de_vitesse_de_l_enonce_sont_publies() -> None:
    assert MACROS["ChampBU"] == "t x"
    assert MACROS["ChampBV"] == "x - y"
    assert MACROS["ChampCU"] == "2 x"
    assert MACROS["ChampCV"] == "- 2 y"


def test_le_rapport_de_hauteur_prismatique_est_derive() -> None:
    assert MACROS["RapportHauteurPrismatique"] == r"\frac{1}{2}"


def test_la_pente_des_lignes_de_courant_de_c_est_derivee() -> None:
    assert MACROS["PenteLignesCourantC"] == r"- \frac{y}{x}"
