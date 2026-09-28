"""Les reponses du corrige de la partie pratique, recalculees.

Le corrige ecrit `R_4 = 31,845 micro-ohms` puis, deux lignes plus bas,
`32,845`. Les deux ne peuvent pas etre vrais. Ces tests tranchent, comme
ils tranchent le reste.
"""

import math

import pytest
import sympy as sp

from polytech.electronique.domain.amplification import (
    gain_maximal_par_etage,
    gain_non_inverseur,
)
from polytech.electronique.domain.biquad import TowThomas
from polytech.electronique.domain.transfert import S

MICRO, NANO = 1e-6, 1e-9


def test_le_diviseur_capacitif_charge_a_le_transfert_du_corrige() -> None:
    """Question 2.1 : C_2 charge le diviseur et deplace le gain continu."""
    r, c1, c2 = sp.symbols("R C_1 C_2", positive=True)
    obtenu = (1 / (S * c1)) / (r + 1 / (S * c1) + 1 / (S * c2))
    attendu = 1 / (S * c1 * r + (1 + c1 / c2))
    assert sp.simplify(obtenu - attendu) == 0


def test_court_circuiter_c2_ramene_au_simple_passe_bas() -> None:
    """Question 2.2 : la masse de la sonde met C_2 hors circuit."""
    r, c1, c2 = sp.symbols("R C_1 C_2", positive=True)
    charge = 1 / (S * c1 * r + (1 + c1 / c2))
    sans_c2 = sp.limit(charge, c2, sp.oo)
    assert sp.simplify(sans_c2 - 1 / (S * c1 * r + 1)) == 0


def test_la_reponse_indicielle_est_bien_celle_du_corrige() -> None:
    """Question 3.1 : decomposition en elements simples puis Laplace inverse."""
    a, b, c, t = sp.symbols("a b c t", real=True)
    obtenue = sp.inverse_laplace_transform(a / ((S - b) * (S - c)), S, t)
    attendue = a / (b - c) * (sp.exp(b * t) - sp.exp(c * t))
    # Sans `expand` : developper les exponentielles empeche la
    # simplification de reconnaitre qu'elles s'annulent.
    assert sp.simplify(obtenue - attendue * sp.Heaviside(t)) == 0


def test_un_seul_etage_exigerait_75_megahertz() -> None:
    """Question 4.1 : c'est ce qui impose deux etages."""
    assert pytest.approx(75e6) == 1500 * 50e3


def test_les_deux_gains_maximaux_laissent_passer_1500() -> None:
    g1 = gain_maximal_par_etage(1e6, 50e3)
    g2 = gain_maximal_par_etage(4e6, 50e3)
    assert (g1, g2) == (pytest.approx(20.0), pytest.approx(80.0))
    assert g1 * g2 == pytest.approx(1600.0)
    assert g1 * g2 > 1500


def test_les_resistances_choisies_donnent_exactement_1500() -> None:
    """Question 4.1 : 1 k / 17,75 k puis 1 k / 79 k."""
    premier = gain_non_inverseur(1e3, 17.75e3)
    second = gain_non_inverseur(1e3, 79e3)
    assert premier == pytest.approx(18.75)
    assert second == pytest.approx(80.0)
    assert premier * second == pytest.approx(1500.0)


def test_chaque_etage_reste_sous_son_gain_maximal() -> None:
    """La verification que le corrige ne fait pas : le partage tient-il."""
    assert gain_non_inverseur(1e3, 17.75e3) <= gain_maximal_par_etage(1e6, 50e3)
    assert gain_non_inverseur(1e3, 79e3) <= gain_maximal_par_etage(4e6, 50e3)


def test_le_prototype_de_tow_thomas_donne_les_valeurs_normalisees() -> None:
    """Question 6.1 : omega_0 = 15,70 krad/s, Q = 1/racine(2), gain 2."""
    cellule = TowThomas(pulsation=15.70e3, qualite=1 / math.sqrt(2), gain=2.0)
    assert cellule.r1_normalisee() == pytest.approx(45.04 * MICRO, rel=1e-3)
    assert cellule.r2_normalisee() == pytest.approx(63.69 * MICRO, rel=1e-3)
    assert cellule.r4_normalisee() == pytest.approx(31.845 * MICRO, rel=1e-4)


def test_la_seconde_ecriture_de_r4_est_la_fautive() -> None:
    """31,845 et 32,845 ne peuvent pas valoir tous deux 100 ohms.

    Le corrige annonce `R_4 = 32,845 micro-ohms * gamma_0 = 100 ohms`. Avec
    32,845 le produit vaut 103 ohms ; avec 31,845, il vaut 100. C'est donc
    le premier chiffre qui etait juste, et la ligne finale porte un lapsus.
    """
    facteur = 3.14e6
    assert 31.845 * MICRO * facteur == pytest.approx(100.0, rel=1e-3)
    assert 32.845 * MICRO * facteur != pytest.approx(100.0, rel=1e-3)


def test_le_facteur_d_echelle_respecte_les_deux_contraintes() -> None:
    """Il faut a la fois des resistances >= 100 ohms et des capacites <= 1 uF."""
    cellule = TowThomas(pulsation=15.70e3, qualite=1 / math.sqrt(2), gain=2.0)
    facteur = cellule.facteur_d_echelle(
        resistance_minimale=100.0,
        capacite_maximale=1e-6,
    )
    assert facteur == pytest.approx(3.14e6, rel=1e-2)
    assert cellule.r4_normalisee() * facteur >= 100.0
    assert 1.0 / facteur <= 1e-6


def test_les_valeurs_finales_sont_celles_du_corrige() -> None:
    cellule = TowThomas(pulsation=15.70e3, qualite=1 / math.sqrt(2), gain=2.0)
    facteur = 3.14e6
    assert cellule.r1_normalisee() * facteur == pytest.approx(141.43, rel=1e-3)
    assert cellule.r2_normalisee() * facteur == pytest.approx(200.0, rel=1e-3)
    assert cellule.r4_normalisee() * facteur == pytest.approx(100.0, rel=1e-3)
    assert 1.0 / facteur == pytest.approx(318.47 * NANO, rel=1e-3)


def test_deux_resistances_egales_en_parallele_en_valent_la_moitie() -> None:
    """Question 6.2 : 200 ohms en parallele de 200 ohms donnent 100 ohms."""
    assert pytest.approx(100.0) == 1 / (1 / 200 + 1 / 200)
