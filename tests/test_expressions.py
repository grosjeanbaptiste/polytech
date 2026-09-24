"""Les resultats symboliques du corrige sont derives, pas transcrits."""

from collections.abc import Mapping

from fluides.application.formatage import expression_latex
from fluides.application.generation import GenererMacros
from fluides.domain.cinematique.symboles import T, X


class _EnMemoire:
    def __init__(self) -> None:
        self.macros: dict[str, str] = {}

    def publier(self, macros: Mapping[str, str]) -> None:
        self.macros = dict(macros)


def _macros() -> dict[str, str]:
    publication = _EnMemoire()
    GenererMacros(publication=publication).executer()
    return publication.macros


def test_le_formatage_produit_du_latex_mathematique() -> None:
    assert expression_latex(X**2) == "x^{2}"
    assert expression_latex(T / 2) == r"\frac{t}{2}"


def test_la_divergence_de_b_est_publiee_symboliquement() -> None:
    assert _macros()["DivergenceB"] == "t - 1"


def test_le_taux_de_rotation_de_b_est_publie_en_fraction_latex() -> None:
    assert _macros()["TauxRotationB"] == r"\frac{1}{2}"


def test_le_champ_d_acceleration_de_b_est_publie() -> None:
    macros = _macros()
    assert macros["ChampAccelerationBX"] == r"x \left(t^{2} + 1\right)"
    assert macros["ChampAccelerationBY"] == "t x - x + y"


def test_la_fonction_de_courant_et_le_potentiel_de_c_sont_publies() -> None:
    macros = _macros()
    assert macros["FonctionCourantC"] == "2 x y"
    assert macros["PotentielC"] == "x^{2} - y^{2}"
