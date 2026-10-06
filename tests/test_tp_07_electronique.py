"""TP 7 d'electronique : biquads, filtres LC en echelle et filtres actifs.

Oracles : le corrige officiel (`ElectroniqueInstrumentation/TP/
TP_7_Solution.pdf`, douze pages), dont chaque valeur est citee avec sa
page. La bibliotheque ne les repete pas : chaque filtre dimensionne est
remonte en circuit -- matrices de chaine pour les echelles, analyse
nodale pour les cellules actives -- et c'est le circuit qui dit sa
coupure, son ondulation ou son facteur de qualite.

Les ecarts que ces recoupements etablissent, tous d'arrondi sauf deux :

* page 3 : `racine(1 + racine(2))` imprime 1.558, vaut 1,5538 ;
* page 10 : la borne des capacites est ecrite « 100 uF » deux fois, alors
  que l'enonce dit 10 uF et que le calcul qui suit emploie bien 10 uF ;
* pages 8, 10 et 12 : `2 pi f` arrondi a trois chiffres avant de diviser,
  d'ou 40,94 nH pour 41,05, 15,87 nF pour 15,92 et 15,90 nF pour 15,92.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.application import biquad_tp_07 as bq
from polytech.electronique.application import echelles_tp_07 as ec
from polytech.electronique.application import gain_tp_07 as ga
from polytech.electronique.application import sallen_key_tp_07 as sk
from polytech.electronique.domain.methodes_sallen_key import Methode
from polytech.electronique.domain.reponse_de_chebyshev import (
    coupure_a_trois_decibels,
    module_de_chebyshev,
)

S = sp.Symbol("s")
MEGA = 2 * math.pi * 1e6
DIXIEME = 10 ** (-0.1 / 20)
"""Un dixieme de decibel de perte, en amplitude."""


# --- Exercice 2.1 -----------------------------------------------------


def test_ex21_les_limites_du_transfert() -> None:
    """Page 1 : 1 en `s = 0`, 0 ailleurs quand `Q -> 0` ; `w0^2/(s^2 + w0^2)`."""
    a_l_origine, a_qualite_nulle, sans_amortissement = bq.limites_du_transfert()
    assert a_l_origine == 1
    assert a_qualite_nulle == 0
    assert sp.simplify(sans_amortissement - bq.W0**2 / (bq.S**2 + bq.W0**2)) == 0


def test_ex21_les_poles_en_fonction_de_l_amortissement() -> None:
    """Pages 1 et 2 : `w0 (-xi +- j racine(1 - xi^2))`, ou `-xi +- racine(xi^2 - 1)`."""
    complexes = bq.poles_reduits(0.5)
    assert sorted(p.imag for p in complexes) == pytest.approx(
        [-math.sqrt(0.75), math.sqrt(0.75)]
    )
    assert [p.real for p in complexes] == pytest.approx([-0.5, -0.5])
    reels = sorted(p.real for p in bq.poles_reduits(1.25))
    assert reels == pytest.approx([-2.0, -0.5])
    assert all(p.imag == 0 for p in bq.poles_reduits(1.25))
    for fautif in (0.0, -0.5):
        with pytest.raises(ValueError, match="amortissement nul ou negatif"):
            bq.poles_reduits(fautif)


def test_ex21_les_poles_aux_deux_limites() -> None:
    """Page 2 : `+- j w0` quand `Q -> infini` ; l'origine et l'infini quand `Q -> 0`."""
    lents = sorted(p.real for p in bq.poles_reduits(1e4))
    assert lents[1] == pytest.approx(-1 / (2 * 1e4), rel=1e-6)
    assert lents[0] == pytest.approx(-2 * 1e4, rel=1e-6)
    libres = bq.poles_reduits(1e-9)
    assert sorted(p.imag for p in libres) == pytest.approx([-1.0, 1.0])
    assert all(abs(p.real) < 1e-8 for p in libres)


def test_ex21_le_point_de_la_figure() -> None:
    """Page 4 : `xi = 0,02`, 1552,387 rad/s, -2,9922 dB ; la coupure est a 1553,3."""
    assert bq.gain_en_decibels(1552.387, 0.02) == pytest.approx(-2.9922, abs=5e-5)
    coupure = bq.coupure_de_la_figure()
    assert bq.gain_en_decibels(coupure, 0.02) == pytest.approx(
        -10 * math.log10(2), abs=1e-9
    )
    assert coupure == pytest.approx(1553.3, abs=0.05)


def test_ex21_les_courbes_de_la_figure() -> None:
    """Cinq courbes, de 100 a 10 000 rad/s ; chacune part de 0 dB."""
    for xi in bq.AMORTISSEMENTS:
        pulsations, gains = bq.courbe(xi)
        assert (pulsations[0], pulsations[-1]) == pytest.approx((100.0, 10_000.0))
        assert gains[0] == pytest.approx(0.0, abs=0.1)
        assert gains[-1] == pytest.approx(-40.0, abs=0.2)
        resonant = xi < 1 / math.sqrt(2)
        sommet = 1 / (2 * xi * math.sqrt(1 - xi**2)) if resonant else 1.0
        assert max(gains) == pytest.approx(20 * math.log10(sommet), abs=0.1)
    assert bq.AMORTISSEMENTS == (0.02, 0.1, 0.3, 0.5, 1.0)


# --- Exercice 3.1 -----------------------------------------------------


def test_ex31_le_butterworth_d_ordre_deux() -> None:
    """Page 5 : `L1 = 1.12 uH`, `C2 = 450 pF`."""
    filtre = ec.butterworth_dix_megahertz()
    inductance, capacite = filtre.branches
    assert filtre.serie_d_abord
    assert inductance.element()[0] == pytest.approx(1.12e-6, abs=6e-9)
    assert capacite.element()[1] == pytest.approx(450e-12, abs=0.5e-12)
    assert inductance.inductance == pytest.approx(50 * math.sqrt(2) / (10 * MEGA))
    assert capacite.capacite == pytest.approx(math.sqrt(2) / 50 / (10 * MEGA))


def test_ex31_le_circuit_coupe_a_dix_megahertz() -> None:
    """Le circuit lui-meme : la moitie en continu, 3 dB de moins a 10 MHz."""
    filtre = ec.butterworth_dix_megahertz()
    assert (filtre.source, filtre.charge) == (50.0, 50.0)
    assert filtre.module(1e-3) == pytest.approx(0.5, rel=1e-9)
    assert filtre.module(10 * MEGA) == pytest.approx(0.5 / math.sqrt(2))


def test_ex31_seul_un_ordre_impair_egalise_source_et_charge() -> None:
    """Page 6 : `g_(N+1)` vaut 1 aux ordres impairs, 1,3554 aux ordres pairs."""
    charges = ec.charges_de_chebyshev(5)
    assert sorted(charges) == [1, 2, 3, 4, 5]
    assert [charges[n] for n in (1, 3, 5)] == pytest.approx([1.0, 1.0, 1.0])
    assert charges[2] == pytest.approx(1.3554, abs=5e-5)
    assert charges[4] == pytest.approx(charges[2])


# --- Exercice 3.2 -----------------------------------------------------


def test_ex32_le_butterworth_d_ordre_trois() -> None:
    """Page 6 : 312,5 nF, 640 uH, 312,5 nF -- une seule inductance."""
    filtre = ec.passe_bas_d_ordre_trois(chebyshev=False)
    c1, l2, c3 = filtre.branches
    assert not filtre.serie_d_abord
    assert (c1.capacite, c3.capacite) == pytest.approx((312.5e-9, 312.5e-9))
    assert l2.inductance == pytest.approx(640e-6)
    assert (filtre.source, filtre.charge) == (32.0, 32.0)
    assert filtre.module(1e5) == pytest.approx(0.5 / math.sqrt(2))


def test_ex32_le_chebyshev_d_ordre_trois() -> None:
    """Page 7 : 322,375 nF et 367,168 uH, d'apres la table a quatre decimales."""
    filtre = ec.passe_bas_d_ordre_trois(chebyshev=True)
    c1, l2, c3 = filtre.branches
    assert c1.capacite == pytest.approx(322.375e-9, rel=1e-4)
    assert l2.inductance == pytest.approx(367.168e-6, rel=1e-4)
    assert c3.capacite == pytest.approx(c1.capacite)
    assert 2 * filtre.module(1e5) == pytest.approx(DIXIEME, rel=1e-5)
    assert 2 * filtre.module(1e-3) == pytest.approx(1.0, rel=1e-9)


def test_ex32_la_coupure_a_trois_decibels_du_chebyshev() -> None:
    """`w_c` borne l'ondulation, pas les 3 dB : ceux-ci sont 1,389 fois plus haut."""
    rapport = ec.rapport_de_coupure_du_chebyshev(3)
    assert rapport == pytest.approx(1.389, abs=5e-4)
    filtre = ec.passe_bas_d_ordre_trois(chebyshev=True)
    # 1e-4 : la table du cours arrondit `40 / ln 10` a 17,37.
    assert 2 * filtre.module(rapport * 1e5) == pytest.approx(1 / math.sqrt(2), rel=1e-4)
    epsilon = math.sqrt(10**0.01 - 1)
    assert coupure_a_trois_decibels(3, epsilon) == pytest.approx(rapport)
    assert module_de_chebyshev(rapport, 3, epsilon) == pytest.approx(1 / math.sqrt(2))
    assert coupure_a_trois_decibels(2, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="ondulation superieure a 3 dB"):
        coupure_a_trois_decibels(3, 1.5)
    with pytest.raises(ValueError, match="ordre inferieur a un"):
        coupure_a_trois_decibels(0, 0.5)


# --- Exercice 3.3 -----------------------------------------------------


def test_ex33_le_passe_bande() -> None:
    """Page 8 : 40,94 nH, 765 fF, 1,72 nH, 18,21 pF -- a un arrondi pres."""
    filtre = ec.passe_bande()
    serie, parallele, derniere = filtre.branches
    assert filtre.serie_d_abord
    assert [b.serie for b in filtre.branches] == [True, False, True]
    obtenus = (
        serie.inductance,
        serie.capacite,
        parallele.inductance,
        parallele.capacite,
    )
    assert obtenus == pytest.approx((40.94e-9, 765e-15, 1.72e-9, 18.21e-12), rel=5e-3)
    assert (derniere.inductance, derniere.capacite) == pytest.approx(
        (serie.inductance, serie.capacite)
    )


def test_ex33_les_valeurs_sans_arrondi_intermediaire() -> None:
    """`L = 50 g1 / B`, `C = B / (w0^2 50 g1)` ; puis leurs duaux en `g2`."""
    serie, parallele, _ = ec.passe_bande().branches
    g1, g2 = 1.0316, 1.1474
    w0, bande = 900 * MEGA, 200 * MEGA
    assert serie.inductance == pytest.approx(50 * g1 / bande, rel=1e-4)
    assert serie.capacite == pytest.approx(bande / (w0**2 * 50 * g1), rel=1e-4)
    assert parallele.inductance == pytest.approx(50 * bande / (w0**2 * g2), rel=1e-4)
    assert parallele.capacite == pytest.approx(g2 / (50 * bande), rel=1e-4)
    assert serie.inductance is not None
    assert serie.inductance * 1e9 == pytest.approx(41.05, abs=5e-3)
    assert abs(serie.inductance * 1e9 - 40.94) > 0.1


def test_ex33_chaque_branche_resonne_a_la_frequence_centrale() -> None:
    """`L C = 1 / w0^2` pour les trois branches : c'est la transformation."""
    for branche in ec.passe_bande().branches:
        assert branche.inductance is not None
        assert branche.capacite is not None
        resonance = 1 / math.sqrt(branche.inductance * branche.capacite)
        assert resonance == pytest.approx(900 * MEGA)


def test_ex33_la_bande_reelle_n_est_pas_celle_de_l_enonce() -> None:
    """Centre a 900 MHz et largeur de 200 : les bords sont a 805,5 et 1005,5 MHz."""
    bas, haut = ec.bords_de_la_bande()
    assert (bas / MEGA, haut / MEGA) == pytest.approx((805.5, 1005.5), abs=0.05)
    assert math.sqrt(bas * haut) == pytest.approx(900 * MEGA)
    filtre = ec.passe_bande()
    for bord in (bas, haut):
        assert 2 * filtre.module(bord) == pytest.approx(DIXIEME, rel=1e-4)
    assert 2 * filtre.module(900 * MEGA) == pytest.approx(1.0, rel=1e-6)
    assert 2 * filtre.module(800 * MEGA) < DIXIEME * 0.99
    assert ec.perte_a_huit_cents_megahertz() == pytest.approx(
        -20 * math.log10(2 * filtre.module(800 * MEGA))
    )


def test_ex33_le_centre_geometrique_rend_la_bande_demandee() -> None:
    """`f_0 = racine(800 x 1000) = 894,4 MHz` : les bords tombent sur 800 et 1000."""
    centre = ec.centre_geometrique()
    assert centre / MEGA == pytest.approx(894.43, abs=5e-3)
    bas, haut = ec.bords_de_la_bande(centre)
    assert (bas / MEGA, haut / MEGA) == pytest.approx((800.0, 1000.0))
    filtre = ec.passe_bande(centre)
    for bord in (800 * MEGA, 1000 * MEGA):
        assert 2 * filtre.module(bord) == pytest.approx(DIXIEME, rel=1e-4)


# --- Exercice 3.4 -----------------------------------------------------


def test_ex34_le_passe_haut() -> None:
    """Page 9 : `C1 = 34.33 pF`, `L2 = 116.32 nH`, `R_L = 67.77 ohm`."""
    filtre = ec.passe_haut()
    capacite, inductance = filtre.branches
    assert filtre.serie_d_abord
    assert capacite.element()[1] == pytest.approx(34.33e-12, abs=0.02e-12)
    assert inductance.element()[0] == pytest.approx(116.32e-9, abs=0.02e-9)
    assert filtre.charge == pytest.approx(67.77, abs=5e-3)
    assert filtre.source == 50.0


def test_ex34_le_circuit_tient_son_ondulation() -> None:
    """Ordre pair : le gain a l'infini est au creux, comme a la coupure."""
    filtre = ec.passe_haut()
    coupure = 110 * MEGA
    loin = filtre.module(1e6 * coupure)
    assert loin == pytest.approx(filtre.charge / (50 + filtre.charge), rel=1e-6)
    assert filtre.module(coupure) == pytest.approx(loin, rel=1e-4)
    sommet = filtre.module(coupure * math.sqrt(2))
    assert sommet / loin == pytest.approx(1 / DIXIEME, rel=1e-4)
    assert sommet == pytest.approx(math.sqrt(filtre.charge / 50) / 2, rel=1e-4)
    assert filtre.module(coupure / 10) < loin / 25


def test_ex34_l_autre_topologie_demande_une_autre_charge() -> None:
    """Page 9 : `g_3` devient une conductance, 0,7378 ohm normalise -- 36,89 ohm."""
    filtre = ec.passe_haut(prototype_en_serie=False)
    inductance, capacite = filtre.branches
    assert not filtre.serie_d_abord
    assert inductance.element()[1] is None
    assert capacite.element()[0] is None
    assert filtre.charge == pytest.approx(50 * 0.7378, abs=5e-3)
    coupure = 110 * MEGA
    loin = filtre.module(1e6 * coupure)
    assert filtre.module(coupure) == pytest.approx(loin, rel=1e-4)
    assert filtre.module(coupure * math.sqrt(2)) / loin == pytest.approx(
        1 / DIXIEME, rel=1e-4
    )


# --- Exercice 4.1 -----------------------------------------------------


def test_ex41_la_methode_et_les_deux_bornes_du_facteur_d_echelle() -> None:
    """Page 10 : methode 4, `gamma_0 = 100`, et 0,159 pour les capacites."""
    assert sk.METHODE_DU_4_1 is Methode.SENSIBILITE_MINIMALE
    assert sk.facteur_pour_les_resistances() == pytest.approx(100.0)
    assert sk.facteur_pour_les_capacites() == pytest.approx(0.159, abs=5e-4)
    assert sk.facteur_pour_les_capacites() == pytest.approx(
        2 / (2 * math.pi * 200e3 * 10e-6)
    )
    assert sk.facteur_pour_les_capacites(100e-6) == pytest.approx(0.0159, abs=5e-5)


def test_ex41_la_cellule() -> None:
    """Page 10 : 15,87 nF et 3,968 nF, avec `w_c` arrondi a `1.26 10^6` rad/s."""
    cellule = sk.cellule_du_4_1()
    assert (float(cellule.r1), float(cellule.r2)) == pytest.approx((100.0, 100.0))
    assert float(cellule.c1) == pytest.approx(15.87e-9, rel=5e-3)
    assert float(cellule.c2) == pytest.approx(3.968e-9, rel=5e-3)
    assert float(cellule.c1) == pytest.approx(2 / (2 * math.pi * 200e3 * 100))
    assert float(cellule.c1) * 1e9 == pytest.approx(15.92, abs=5e-3)
    assert float(cellule.c1 / cellule.c2) == pytest.approx(4.0)
    assert cellule.gain == 1
    assert float(cellule.c1) < sk.CAPACITE_MAXIMALE_DU_4_1


def test_ex41_le_circuit_a_bien_la_pulsation_et_la_qualite_visees() -> None:
    """L'analyse nodale : `|H(j w_c)| = Q = 1`, et une phase de -90 degres."""
    cellule = sk.cellule_du_4_1()
    reponse = complex(cellule.transfert(sp.I * 2 * sp.pi * 200_000))
    assert abs(reponse) == pytest.approx(1.0, rel=1e-9)
    assert math.degrees(math.atan2(reponse.imag, reponse.real)) == pytest.approx(-90.0)
    assert complex(cellule.transfert(sp.Integer(0))) == pytest.approx(1.0)


def test_ex41_la_cellule_a_dix_kilohms() -> None:
    """Page 11 : 158,7 pF et 39,68 pF, au meme arrondi."""
    cellule = sk.cellule_du_4_1(resistance=10e3)
    assert float(cellule.r1) == pytest.approx(1e4)
    assert float(cellule.c1) == pytest.approx(158.7e-12, rel=5e-3)
    assert float(cellule.c2) == pytest.approx(39.68e-12, rel=5e-3)
    origine = sk.cellule_du_4_1()
    for frequence in (50e3, 200e3, 700e3):
        point = 2j * math.pi * frequence
        assert complex(cellule.transfert(point)) == pytest.approx(
            complex(origine.transfert(point))
        )


def test_ex41_la_vraie_coupure_et_la_cascade_de_trois_cellules() -> None:
    """`Q = 1` : la cellule coupe a 254 kHz, la cascade a 220 kHz, bosse de 3,7 dB."""
    assert sk.coupure_reelle_du_4_1() == pytest.approx(254.4e3, abs=50)
    assert sk.coupure_reelle_du_4_1(cellules=3) == pytest.approx(220.4e3, abs=50)
    assert sk.bosse_du_4_1(cellules=1) == pytest.approx(1.249, abs=5e-4)
    assert sk.bosse_du_4_1(cellules=3) == pytest.approx(3.748, abs=5e-4)
    cellule = sk.cellule_du_4_1()
    point = 2j * math.pi * sk.coupure_reelle_du_4_1(cellules=3)
    cascade = abs(complex(cellule.transfert(point))) ** 3
    assert cascade == pytest.approx(1 / math.sqrt(2), rel=1e-9)


def test_ex41_un_ordre_six_sans_bosse() -> None:
    """Trois cellules de `Q` differents -- 0,518, 0,707, 1,932 : un Butterworth."""
    cellules = sk.cellules_de_butterworth()
    qualites = [float(c.qualite()) for c in cellules]
    assert qualites == pytest.approx([1 / 1.932, 1 / 1.414, 1 / 0.518], abs=2e-3)
    assert all(float(c.r1) == pytest.approx(100.0) for c in cellules)
    assert all(float(c.c1) < sk.CAPACITE_MAXIMALE_DU_4_1 for c in cellules)
    for rapport in (0.5, 1.0, 2.0):
        point = 2j * math.pi * 200e3 * rapport
        module = math.prod(abs(complex(c.transfert(point))) for c in cellules)
        assert module == pytest.approx(1 / math.sqrt(1 + rapport**12), rel=1e-9)
    assert float(cellules[-1].c1) * 1e9 == pytest.approx(30.75, abs=5e-3)


def test_ex41_les_sensibilites_des_quatre_methodes() -> None:
    """A `Q = 1` : la pire sensibilite passive vaut 3/2, 3/2, 1,08 et 1/2."""
    pires = sk.pires_sensibilites_passives(1.0)
    assert pires[Methode.ELEMENTS_EGAUX] == pytest.approx(1.5)
    assert pires[Methode.CAPACITES_EGALES] == pytest.approx(1.5)
    assert pires[Methode.SENSIBILITE_MODEREE] == pytest.approx(0.5 + 1 / math.sqrt(3))
    assert pires[Methode.SENSIBILITE_MINIMALE] == pytest.approx(0.5)
    a_cinq = sk.pires_sensibilites_passives(5.0)
    assert a_cinq[Methode.SENSIBILITE_MINIMALE] == pytest.approx(0.5)
    assert a_cinq[Methode.CAPACITES_EGALES] == pytest.approx(25.5)


# --- Exercice 4.2 -----------------------------------------------------


def test_ex42_le_gain_et_la_seule_methode_possible() -> None:
    """Pages 11 et 12 : `G = 5/2`, methode 1, `Q = 2`."""
    assert ga.gain() == pytest.approx(2.5)
    methode, qualite = ga.methode()
    assert methode is Methode.ELEMENTS_EGAUX
    assert qualite == pytest.approx(2.0)
    with pytest.raises(ValueError, match="pas seule a tenir le gain"):
        ga.methode(2.0)
    with pytest.raises(ValueError, match="gain de trois ou plus"):
        ga.methode(3.5)
    assert ga.methode(1.5)[1] == pytest.approx(1 / 1.5)


def test_ex42_le_facteur_d_echelle_et_les_composants() -> None:
    """Page 12 : `15.9 uohm`, `gamma_0 >= 6.289 10^7` et `10^7`, 1 kohm, 15,90 nF."""
    assert ga.resistance_normalisee() * 1e6 == pytest.approx(15.9, abs=0.05)
    resistances, capacites = ga.facteurs_d_echelle()
    assert resistances == pytest.approx(6.289e7, rel=2e-3)
    assert resistances == pytest.approx(1000 * 2 * math.pi * 10e3)
    assert capacites == pytest.approx(1e7)
    cellule = ga.cellule()
    assert (float(cellule.r1), float(cellule.r2)) == pytest.approx((1000.0, 1000.0))
    assert float(cellule.c1) == pytest.approx(15.90e-9, rel=2e-3)
    assert float(cellule.c1) * 1e9 == pytest.approx(15.92, abs=5e-3)
    assert float(cellule.c2) == pytest.approx(float(cellule.c1))
    assert float(cellule.gain) == pytest.approx(2.5)
    assert float(cellule.c1) <= ga.CAPACITE_MAXIMALE


def test_ex42_le_circuit_a_bien_le_gain_et_la_qualite_visees() -> None:
    """L'analyse nodale : 2,5 en continu, et `G Q = 5` a 10 kHz."""
    cellule = ga.cellule()
    assert complex(cellule.transfert(sp.Integer(0))) == pytest.approx(2.5)
    a_dix = complex(cellule.transfert(sp.I * 2 * sp.pi * 10_000))
    assert abs(a_dix) == pytest.approx(5.0, rel=1e-9)


def test_ex42_le_pont_de_gain() -> None:
    """Page 12 : 1 kohm et 1,5 kohm ; compense en courants, 3,33 kohm et 5 kohm."""
    assert ga.pont_officiel() == pytest.approx((1000.0, 1500.0))
    vers_la_masse, retour = ga.pont_compense()
    assert (vers_la_masse, retour) == pytest.approx((10_000 / 3, 5000.0))
    assert 1 + retour / vers_la_masse == pytest.approx(2.5)
    parallele = vers_la_masse * retour / (vers_la_masse + retour)
    assert parallele == pytest.approx(2000.0)
    officiel = ga.pont_officiel()
    assert officiel[0] * officiel[1] / sum(officiel) == pytest.approx(600.0)


def test_ex42_la_bande_plate_et_la_surtension() -> None:
    """`Q = 2` : gain plat a 1 dB jusqu'a 3,56 kHz, sommet de 5,16 a 9,35 kHz."""
    assert ga.bande_plate() == pytest.approx(3559.0, abs=1.0)
    frequence, gain = ga.sommet()
    assert frequence == pytest.approx(9354.0, abs=1.0)
    assert gain == pytest.approx(5.164, abs=5e-4)
    cellule = ga.cellule()
    au_sommet = abs(complex(cellule.transfert(2j * math.pi * frequence)))
    assert au_sommet == pytest.approx(gain, rel=1e-9)
    assert ga.crete_au_sommet() == pytest.approx(10.33, abs=5e-3)
    assert ga.crete_au_sommet() > ga.ALIMENTATION


def test_ex42_la_vraie_coupure() -> None:
    """`Q = 2` : -3 dB a 14,85 kHz ; pour couper a 10 kHz, `f_0 = 6,74 kHz`."""
    assert ga.coupure_reelle() == pytest.approx(14_845.0, abs=5.0)
    assert ga.frequence_propre_pour_couper_a_dix_kilohertz() == pytest.approx(
        6736.0, abs=5.0
    )
    cellule = ga.cellule()
    point = 2j * math.pi * ga.coupure_reelle()
    assert abs(complex(cellule.transfert(point))) == pytest.approx(2.5 / math.sqrt(2))
