"""Couche application : resultats de l'activite et generation des macros."""

from collections.abc import Mapping

import pytest

from polytech.fluides.application.activite import ResultatsDeLActivite
from polytech.fluides.application.generation import GenererMacros

RESULTATS = ResultatsDeLActivite().executer()


def test_la_requete_restitue_la_pression_de_l_air() -> None:
    assert RESULTATS.pression_air_kpa == pytest.approx(129.6469, abs=1e-4)
    assert RESULTATS.pression_relative_kpa == pytest.approx(44.0469, abs=1e-4)


def test_la_requete_restitue_les_trois_contributions() -> None:
    assert RESULTATS.contributions_pa == pytest.approx([981.0, 1667.7, 46695.6])
    assert RESULTATS.part_mercure_pourcent == pytest.approx(94.63, abs=0.01)


def test_la_requete_restitue_les_rapports_du_barrage() -> None:
    assert RESULTATS.rapport_poussee_prismatique == pytest.approx(0.25)
    assert RESULTATS.rapport_moment_prismatique == pytest.approx(0.125)
    assert RESULTATS.rapport_poussee_vallee_v == pytest.approx(0.5)


def test_la_requete_restitue_l_acceleration_de_l_enonce_b() -> None:
    assert RESULTATS.acceleration_b == pytest.approx((5.0, 3.0))
    assert RESULTATS.norme_acceleration_b == pytest.approx(5.8310, abs=1e-4)
    assert RESULTATS.angle_acceleration_b_degres == pytest.approx(30.96, abs=0.01)


def test_la_requete_restitue_les_expressions_en_latex() -> None:
    """Les expressions sont derivees par sympy puis rendues en LaTeX."""
    assert RESULTATS.fonction_de_courant_c == "2 x y"
    assert RESULTATS.potentiel_c == "x^{2} - y^{2}"


class _PublicationEnMemoire:
    """Adaptateur de test : retient les macros au lieu de les ecrire."""

    def __init__(self) -> None:
        self.macros: dict[str, str] = {}

    def publier(self, macros: Mapping[str, str]) -> None:
        self.macros = dict(macros)


def test_la_commande_publie_des_nombres_lisibles_par_siunitx() -> None:
    """La mise en forme francaise est le travail de siunitx, pas le notre.

    Publier des nombres bruts evite de dupliquer cette responsabilite entre
    la bibliotheque et le document.
    """
    publication = _PublicationEnMemoire()
    GenererMacros(publication=publication).executer()
    assert publication.macros["PressionAirKPa"] == "129.65"
    assert publication.macros["ContributionMercurePa"] == "46695.6"
    assert publication.macros["PartMercurePourcent"] == "94.6"
    assert publication.macros["ContributionEauKPa"] == "0.981"
    assert publication.macros["PressionAirArrondieKPa"] == "130"


def test_la_commande_refuse_une_publication_absente() -> None:
    with pytest.raises(TypeError):
        GenererMacros()  # type: ignore[call-arg]
