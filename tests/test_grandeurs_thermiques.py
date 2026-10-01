"""Objets-valeur de la thermique : ce qu'ils refusent, et pourquoi.

Deux refus portent une physique, et non une coquetterie de typage.

Une temperature **absolue** negative n'existe pas : le zero kelvin est une
borne, pas une convention. Or l'enonce du TP 1 donne ses temperatures en
degres Celsius pour le charbon et en kelvins pour l'hydrogene, dans le
meme document. Melanger les deux echelles dans une soustraction donne un
resultat faux de 273,15 sans rien signaler -- d'ou un objet qui porte
l'echelle, et une seule.

Un coefficient d'echange **nul** est au contraire parfaitement legitime :
c'est exactement ce que veut dire « paroi adiabatique ». Le refuser
obligerait a traiter le cas (a) du TP comme un autre probleme, alors qu'il
est le meme avec K = 0.
"""

from __future__ import annotations

import pytest

from polytech.thermique.domain.echanges import (
    CoefficientDEchange,
    DensiteDeFlux,
)
from polytech.thermique.domain.grandeurs import Temperature
from polytech.thermique.domain.materiaux import (
    ChaleurLatente,
    Conductivite,
    Epaisseur,
    GenerationVolumique,
)


def test_une_temperature_se_donne_en_celsius_ou_en_kelvins() -> None:
    """Les deux echelles coexistent dans l'enonce : l'objet les reconcilie."""
    assert Temperature.en_celsius(20.0).kelvins == pytest.approx(293.15)
    assert Temperature.en_kelvins(20.0).celsius == pytest.approx(-253.15)


def test_une_temperature_absolue_negative_est_refusee() -> None:
    """Sous le zero kelvin il n'y a pas de physique, mais une faute d'echelle."""
    with pytest.raises(ValueError, match="temperature absolue negative"):
        Temperature.en_kelvins(-1.0)


def test_un_ecart_de_temperature_ne_depend_pas_de_l_echelle() -> None:
    """20 degres d'ecart valent 20 K : c'est ce qui rend la soustraction sure."""
    chaud = Temperature.en_celsius(30.0)
    froid = Temperature.en_celsius(10.0)
    assert chaud.ecart_a(froid).kelvins == pytest.approx(20.0)


def test_un_coefficient_d_echange_nul_est_la_paroi_adiabatique() -> None:
    """Le cas (a) du TP 1 n'est pas un autre probleme : c'est K = 0."""
    assert CoefficientDEchange(watts_par_m2_k=0.0).est_adiabatique


def test_un_coefficient_d_echange_negatif_est_refuse() -> None:
    """Il ferait remonter la chaleur le long de son propre gradient."""
    with pytest.raises(ValueError, match="coefficient d'echange negatif"):
        CoefficientDEchange(watts_par_m2_k=-1.0)


@pytest.mark.parametrize(
    ("fabrique", "message"),
    [
        (lambda: Conductivite(watts_par_m_k=0.0), "conductivite nulle"),
        (lambda: Epaisseur(metres=0.0), "epaisseur nulle"),
        (lambda: ChaleurLatente(joules_par_kg=0.0), "chaleur latente nulle"),
    ],
)
def test_les_grandeurs_qui_divisent_refusent_le_zero(
    fabrique: object,
    message: str,
) -> None:
    """Chacune de ces trois est un denominateur quelque part dans le TP."""
    assert callable(fabrique)
    with pytest.raises(ValueError, match=message):
        fabrique()


def test_une_generation_interne_nulle_reste_licite() -> None:
    """Le reservoir d'hydrogene n'a aucune source : q = 0 y est la donnee."""
    assert GenerationVolumique(watts_par_m3=0.0).watts_par_m3 == 0.0


def test_une_densite_de_flux_peut_etre_negative() -> None:
    """Le signe dit le sens : un flux entrant est l'oppose d'un sortant."""
    assert DensiteDeFlux(watts_par_m2=-150.0).watts_par_m2 == pytest.approx(-150.0)


def test_une_valeur_non_finie_est_refusee_partout() -> None:
    """Un `nan` se propagerait dans toutes les macros sans rien casser."""
    with pytest.raises(ValueError, match="valeur non finie"):
        Conductivite(watts_par_m_k=float("inf"))
