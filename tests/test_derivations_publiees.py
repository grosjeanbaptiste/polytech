"""Chaque raisonnement du corrige doit exister comme macro publiee."""

from collections.abc import Mapping

import pytest

from polytech.fluides.application.generation import GenererMacros


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
        ("EquationManometre", r"\rho_{\text{Hg}}"),
        ("PousseeVerticale", "H^{2} L g \\rho"),
        ("MomentVertical", "H^{3} L g \\rho"),
        ("PousseeValleeV", r"\alpha"),
        ("RapportPousseePrismatique", r"\frac{1}{4}"),
        ("RapportMomentPrismatique", r"\frac{1}{8}"),
        ("LoiVolumeB", "e^{"),
        ("TenseurB", "matrix"),
        ("LignesCourantC", "x y = C"),
        ("EcartMaterielCX", "e^{2 t}"),
        ("EcartMaterielCY", "e^{- 2 t}"),
        ("DeriveeUDeT", "x"),
        ("DeriveeVDeY", "-1"),
    ],
)
def test_la_macro_existe_et_porte_la_derivation(macro: str, attendu: str) -> None:
    assert macro in MACROS, f"macro absente : {macro}"
    assert attendu in MACROS[macro]


def test_les_taux_de_deformation_de_c_sont_publies() -> None:
    assert MACROS["EpsXXC"] == "2"
    assert MACROS["EpsYYC"] == "-2"
    assert MACROS["EpsXYC"] == "0"
