"""La coque spherique de silice, et le reservoir d'hydrogene liquide du TP 1.

Ici le corrige officiel s'arrete : sa planche « Calcul des valeurs
numeriques » est une image dont rien ne s'extrait, et la transcription ne
porte donc **aucun chiffre** pour cet exercice. Il n'y a rien a recouper.

C'est exactement pourquoi ces tests calculent le flux par **deux chemins
independants** : par le schema en resistances, comme le corrige l'enseigne,
et en resolvant l'equation de la chaleur en coordonnees spheriques avec ses
deux conditions aux limites. Les deux doivent tomber sur le meme watt.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.thermique.domain.conduction.spherique import CoqueSpherique
from polytech.thermique.domain.conduction.symboles import RAYON
from polytech.thermique.domain.echanges import CoefficientDEchange
from polytech.thermique.domain.geometrie import Rayon
from polytech.thermique.domain.grandeurs import Temperature
from polytech.thermique.domain.materiaux import ChaleurLatente, Conductivite, Epaisseur
from polytech.thermique.domain.reservoir import ReservoirCryogenique

SILICE = Conductivite(watts_par_m_k=0.0017)
INTERIEUR = Rayon(metres=0.3)
COQUE = CoqueSpherique(
    conductivite=SILICE,
    interieur=INTERIEUR,
    exterieur=INTERIEUR.augmente_de(Epaisseur(metres=0.025)),
)
RESERVOIR = ReservoirCryogenique(
    coque=COQUE,
    echange_exterieur=CoefficientDEchange(watts_par_m2_k=8.0),
    temperature_interieure=Temperature.en_kelvins(20.0),
    temperature_exterieure=Temperature.en_kelvins(300.0),
    chaleur_latente=ChaleurLatente(joules_par_kg=4.5e5),
)


def test_le_rayon_exterieur_se_deduit_de_l_epaisseur() -> None:
    """L'enonce donne le diametre interieur et l'epaisseur, pas les deux rayons."""
    assert COQUE.exterieur.metres == pytest.approx(0.325)


def test_une_coque_inversee_est_refusee() -> None:
    """Un rayon exterieur plus petit donnerait une resistance negative."""
    with pytest.raises(ValueError, match="rayon exterieur"):
        CoqueSpherique(
            conductivite=SILICE,
            interieur=Rayon(metres=0.325),
            exterieur=Rayon(metres=0.3),
        )


def test_le_profil_spherique_varie_en_un_sur_r() -> None:
    """Et non lineairement : c'est la conservation du flux qui l'impose.

    En plan la temperature est affine ; en spherique la surface traversee
    croit comme r^2, donc le gradient decroit comme 1/r^2 et le profil
    comme 1/r. Confondre les deux est l'erreur classique.
    """
    profil = COQUE.profil_general()
    # r^2 dT/dr ne depend plus de r : c'est le flux total, constant.
    assert RAYON not in sp.simplify(sp.diff(profil, RAYON) * RAYON**2).free_symbols


def test_la_conservation_du_flux_est_satisfaite_par_le_profil() -> None:
    """d/dr (r^2 dT/dr) = 0, ce dont le profil en 1/r est la solution."""
    loi = COQUE.loi_de_conservation()
    profil = COQUE.profil_general()
    substitue = loi.lhs.subs(sp.Function("T")(RAYON), profil).doit()
    assert sp.simplify(substitue) == 0


def test_la_resistance_de_la_coque_vaut_la_formule_du_cours() -> None:
    """R = (1/r_i - 1/r_e) / (4 pi lambda), soit 12,00 K/W pour la silice."""
    attendue = (1 / 0.3 - 1 / 0.325) / (4 * math.pi * 0.0017)
    assert COQUE.resistance().kelvins_par_watt == pytest.approx(attendue)
    assert COQUE.resistance().kelvins_par_watt == pytest.approx(12.0026, abs=0.0005)


def test_la_silice_porte_presque_toute_la_resistance() -> None:
    """99,2 % : negliger le film exterieur ne couterait que 0,8 % sur le flux."""
    part = RESERVOIR.coque.resistance().kelvins_par_watt
    total = RESERVOIR.resistance_totale().kelvins_par_watt
    assert part / total == pytest.approx(0.9922, abs=0.0005)


def test_le_flux_traverse_les_deux_resistances_en_serie() -> None:
    """280 K a franchir, 12,097 K/W a traverser : 23,15 W."""
    assert RESERVOIR.flux() == pytest.approx(23.147, abs=0.005)


def test_le_meme_flux_sort_de_l_equation_de_la_chaleur() -> None:
    """Le recoupement : aucune formule de resistance, les seules conditions.

    Le schema en resistances est un raccourci enseigne ; cette resolution
    n'en emprunte rien. Deux chemins, un seul watt.
    """
    assert RESERVOIR.flux_par_resolution_directe() == pytest.approx(
        RESERVOIR.flux(),
        rel=1e-9,
    )


def test_la_paroi_exterieure_est_presque_a_l_ambiance() -> None:
    """24,7 degres pour 26,9 d'ambiance : le vase ne givre pas, il chauffe peu."""
    paroi = RESERVOIR.temperature_de_paroi_exterieure()
    assert paroi.kelvins == pytest.approx(297.82, abs=0.01)


def test_le_debit_evapore_vaut_le_flux_divise_par_la_chaleur_latente() -> None:
    """5,14e-5 kg/s, soit 4,44 kg par jour -- l'ordre de grandeur attendu."""
    assert RESERVOIR.debit_evapore() == pytest.approx(5.1437e-5, rel=1e-3)
    assert RESERVOIR.debit_evapore() * 86_400 == pytest.approx(4.444, abs=0.005)


def test_un_reservoir_plus_chaud_dedans_que_dehors_est_refuse() -> None:
    """Le flux changerait de sens et le debit « evapore » serait negatif."""
    with pytest.raises(ValueError, match="plus chaud"):
        ReservoirCryogenique(
            coque=COQUE,
            echange_exterieur=CoefficientDEchange(watts_par_m2_k=8.0),
            temperature_interieure=Temperature.en_kelvins(300.0),
            temperature_exterieure=Temperature.en_kelvins(20.0),
            chaleur_latente=ChaleurLatente(joules_par_kg=4.5e5),
        )
