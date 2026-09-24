"""Cours II.2 (Stevin) et II.3 (manometre multi-fluides)."""

import pytest

from fluides.domain.grandeurs import Hauteur, MasseVolumique, Pression
from fluides.domain.statique.manometre import Manometre, Sens, Troncon
from fluides.domain.statique.stevin import variation_de_pression

EAU = MasseVolumique(kg_par_m3=1_000.0)
HUILE = MasseVolumique(kg_par_m3=850.0)
MERCURE = MasseVolumique(kg_par_m3=13_600.0)


def test_stevin_donne_le_poids_de_la_colonne() -> None:
    delta = variation_de_pression(EAU, Hauteur(metres=0.10))
    assert delta.pascals == pytest.approx(981.0)


def test_stevin_sur_la_colonne_de_mercure_de_l_activite() -> None:
    delta = variation_de_pression(MERCURE, Hauteur(metres=0.35))
    assert delta.pascals == pytest.approx(46_695.6)


def test_le_manometre_multi_fluides_de_l_activite() -> None:
    """Enonce A : reponse attendue p1 = 130 kPa."""
    manometre = Manometre(
        troncons=(
            Troncon(EAU, Hauteur(metres=0.10), Sens.DESCENTE),
            Troncon(HUILE, Hauteur(metres=0.20), Sens.DESCENTE),
            Troncon(MERCURE, Hauteur(metres=0.35), Sens.MONTEE),
        ),
    )
    amont = manometre.pression_amont(aval=Pression(pascals=85_600.0))
    assert amont.pascals == pytest.approx(129_646.9, abs=0.01)
    assert round(amont.kilopascals) == 130


def test_le_manometre_est_reversible() -> None:
    """Remonter le manometre depuis l'amont redonne la pression aval."""
    manometre = Manometre(
        troncons=(Troncon(MERCURE, Hauteur(metres=0.35), Sens.MONTEE),),
    )
    aval = Pression(pascals=85_600.0)
    assert manometre.pression_aval(
        amont=manometre.pression_amont(aval=aval),
    ).pascals == pytest.approx(aval.pascals)


def test_un_manometre_sans_troncon_est_refuse() -> None:
    with pytest.raises(ValueError, match="au moins un troncon"):
        Manometre(troncons=())


def test_la_contribution_de_chaque_troncon_est_consultable() -> None:
    """Sert a l'analyse critique : quel fluide pese dans la mesure."""
    manometre = Manometre(
        troncons=(
            Troncon(EAU, Hauteur(metres=0.10), Sens.DESCENTE),
            Troncon(HUILE, Hauteur(metres=0.20), Sens.DESCENTE),
            Troncon(MERCURE, Hauteur(metres=0.35), Sens.MONTEE),
        ),
    )
    poids = [c.pascals for c in manometre.contributions()]
    assert poids == pytest.approx([981.0, 1_667.7, 46_695.6])
    assert manometre.part_dominante() == pytest.approx(0.9463, abs=1e-4)
