"""Enonce C : efforts hydrostatiques sur un barrage."""

import pytest

from polytech.fluides.domain.grandeurs import Hauteur, MasseVolumique
from polytech.fluides.domain.statique.parements import (
    ParementTriangulaire,
    ParementVertical,
)

EAU = MasseVolumique(kg_par_m3=1_000.0)


def test_la_poussee_sur_un_parement_vertical_varie_comme_le_carre() -> None:
    mur = ParementVertical(fluide=EAU, largeur=10.0)
    assert mur.poussee(Hauteur(metres=10.0)) == pytest.approx(
        0.5 * 1000 * 9.81 * 10 * 100
    )


def test_diviser_la_hauteur_par_deux_divise_la_poussee_par_quatre() -> None:
    mur = ParementVertical(fluide=EAU, largeur=1.0)
    pleine = mur.poussee(Hauteur(metres=10.0))
    demie = mur.poussee(Hauteur(metres=5.0))
    assert demie / pleine == pytest.approx(0.25)


def test_le_moment_de_renversement_est_divise_par_huit() -> None:
    mur = ParementVertical(fluide=EAU, largeur=1.0)
    rapport = mur.moment(Hauteur(metres=5.0)) / mur.moment(Hauteur(metres=10.0))
    assert rapport == pytest.approx(0.125)


def test_le_centre_de_poussee_est_au_tiers_de_la_hauteur() -> None:
    mur = ParementVertical(fluide=EAU, largeur=1.0)
    assert mur.centre_de_poussee(Hauteur(metres=9.0)) == pytest.approx(3.0)


def test_bassin_prismatique_la_hauteur_suit_le_volume() -> None:
    mur = ParementVertical(fluide=EAU, largeur=1.0)
    assert mur.hauteur_pour_fraction_de_volume(0.5) == pytest.approx(0.5)


def test_vallee_en_v_la_poussee_varie_comme_le_cube() -> None:
    vallee = ParementTriangulaire(fluide=EAU, pente=0.5)
    petite = vallee.poussee(Hauteur(metres=5.0))
    grande = vallee.poussee(Hauteur(metres=10.0))
    assert grande / petite == pytest.approx(8.0)


def test_vallee_en_v_halver_le_volume_halve_bien_la_poussee() -> None:
    """La seule geometrie ou l'affirmation du responsable tombe juste."""
    vallee = ParementTriangulaire(fluide=EAU, pente=0.5)
    fraction = vallee.hauteur_pour_fraction_de_volume(0.5)
    reduite = vallee.poussee(Hauteur(metres=10.0 * fraction))
    assert reduite / vallee.poussee(Hauteur(metres=10.0)) == pytest.approx(0.5)


def test_une_largeur_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="largeur"):
        ParementVertical(fluide=EAU, largeur=-1.0)


def test_une_fraction_de_volume_hors_bornes_est_refusee() -> None:
    mur = ParementVertical(fluide=EAU, largeur=1.0)
    with pytest.raises(ValueError, match="fraction"):
        mur.hauteur_pour_fraction_de_volume(1.5)
