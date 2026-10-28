"""Les planches d'electronique : les bruits electroniques (chapitre 4, partie II).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_4_part2.pdf`, 66
planches lues sur leur couche de texte.

Les oracles : `S = 4 k_B T Re[Z]` et sa correction quantique (planches 6 a
9), les impedances bruyantes en serie et en parallele (planches 13 a 16),
`P_w (f_h - f_l + f_c ln(f_h / f_l))` (planche 31), la fiche de l'OPA140
(planche 33 : 8, 5,8 et 5,1 nV/racine(Hz) ; 42 nV efficaces et 250 nV
crete a crete de 0,1 a 10 Hz) et le facteur de bruit (planches 48 a 64).

Deux ecarts de la planche 10 et de la planche 9, d'ordre de grandeur :
1 Mohm sur 10 kHz donne 12,9 uV a 300 K, que la planche arrondit a
« 10 uV » ; et a 50 K, la correction quantique retire deja 22 % a
500 GHz, que la planche dit « no significant effect ».
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain.bruit import (
    densite_spectrale_de_grenaille,
    densite_spectrale_en_un_sur_f,
    densite_spectrale_unilaterale,
    est_blanc,
    puissance_integree_blanc_et_rose,
    tension_efficace_ramenee_a_l_entree,
)
from polytech.electronique.domain.bruit_gaussien import excursion_crete_a_crete
from polytech.electronique.domain.bruit_thermique import (
    densite_de_deux_impedances_en_parallele,
    densite_de_deux_impedances_en_serie,
    densite_thermique,
    densite_thermique_quantique,
    puissance_thermique_totale,
    tension_efficace_thermique,
)
from polytech.electronique.domain.facteur_de_bruit import (
    bruit_dominant_selon_la_source,
    facteur_de_bruit,
    facteur_de_bruit_ponctuel,
    figure_de_bruit,
)
from polytech.electronique.domain.modele_tension_courant import (
    ModeleBlancEtRose,
    densite_ramenee_a_l_entree,
    facteur_de_bruit_du_modele,
    resistance_de_source_optimale,
)
from polytech.thermique.domain.rayonnement.corps_noir import BOLTZMANN, PLANCK

AMBIANTE = 300.0


def test_la_densite_du_bruit_thermique() -> None:
    """Planches 6 et 7 : `4 k_B T R`, et `4 k_B T Re[Z]` pour une impedance."""
    resistance = densite_thermique(1e3, AMBIANTE)
    assert resistance == pytest.approx(
        1.657e-17, rel=1e-3, abs=0
    )  # 4 k T R, 1 kohm a 300 K
    assert densite_thermique(1e3 + 5e3j, AMBIANTE) == resistance
    assert densite_thermique(5e3j, AMBIANTE) == 0.0
    symbolique = densite_spectrale_unilaterale(1e3, AMBIANTE, BOLTZMANN)
    assert resistance == pytest.approx(float(symbolique), rel=1e-9, abs=0)
    with pytest.raises(ValueError, match="partie reelle negative"):
        densite_thermique(-1e3 + 0j, AMBIANTE)


def test_la_correction_quantique_rend_la_puissance_finie() -> None:
    """Planches 8 et 9 : `4 k_B T R x / (e^x - 1)`, `x = h f / k_B T`.

    `int x / (e^x - 1) dx = pi^2 / 6` : `1 / (e^x - 1)` est la somme des
    `e^(-n x)`, et chaque terme s'integre en `1 / n^2`.
    """
    classique = densite_thermique(1e3, AMBIANTE)
    a_un_megahertz = densite_thermique_quantique(1e3, AMBIANTE, 1e6)
    assert a_un_megahertz == pytest.approx(classique, rel=1e-6, abs=0)
    assert densite_thermique_quantique(1e3, AMBIANTE, 0.0) == classique
    a_un_terahertz = densite_thermique_quantique(1e3, AMBIANTE, 1e12) / classique
    assert a_un_terahertz == pytest.approx(0.922, abs=5e-4)
    froid = densite_thermique_quantique(1e3, 50.0, 5e11)
    assert froid / densite_thermique(1e3, 50.0) == pytest.approx(0.779, abs=5e-4)
    x, rang = sp.Symbol("x", positive=True), sp.Symbol("n", positive=True, integer=True)
    terme = sp.integrate(x * sp.exp(-rang * x), (x, 0, sp.oo))
    assert terme == 1 / rang**2
    assert sp.summation(terme, (rang, 1, sp.oo)) == sp.pi**2 / 6
    totale = puissance_thermique_totale(1e3, AMBIANTE)
    reduite = BOLTZMANN * AMBIANTE / PLANCK
    assert totale == pytest.approx(
        classique * reduite * math.pi**2 / 6, rel=1e-9, abs=0
    )
    pas = reduite / 200
    riemann = sum(
        densite_thermique_quantique(1e3, AMBIANTE, (k + 0.5) * pas) * pas
        for k in range(8000)
    )
    assert riemann == pytest.approx(totale, rel=1e-4, abs=0)


def test_le_bruit_thermique_compte_des_microvolts() -> None:
    """Planche 10 : `V_RMS = racine(4 k_B T R B)` ; 1 Mohm sur 10 kHz."""
    efficace = tension_efficace_thermique(1e6, AMBIANTE, 1e4)
    assert efficace == pytest.approx(12.9e-6, abs=5e-8)
    assert efficace**2 == pytest.approx(
        densite_thermique(1e6, AMBIANTE) * 1e4, rel=1e-9, abs=0
    )
    assert 10e-6 < efficace < 15e-6
    with pytest.raises(ValueError, match="bande nulle ou negative"):
        tension_efficace_thermique(1e6, AMBIANTE, 0.0)


def test_deux_impedances_bruyantes_en_serie() -> None:
    """Planche 13 : `4 k_B (T_1 Re[Z_1] + T_2 Re[Z_2])`, sur `Z_1 + Z_2`."""
    premiere, seconde = (1e3 + 2e3j, 300.0), (4e3 - 1e3j, 77.0)
    densite = densite_de_deux_impedances_en_serie(premiere, seconde)
    assert densite == pytest.approx(
        4 * BOLTZMANN * (300 * 1e3 + 77 * 4e3), rel=1e-9, abs=0
    )
    meme = densite_de_deux_impedances_en_serie((1e3, 300.0), (4e3, 300.0))
    assert meme == pytest.approx(densite_thermique(5e3, 300.0), rel=1e-9, abs=0)


def test_deux_resistances_bruyantes_en_parallele() -> None:
    """Planches 14 a 16 : `4 k_B R_1 R_2 (T_1 R_2 + T_2 R_1) / (R_1 + R_2)^2`."""
    densite = densite_de_deux_impedances_en_parallele((1e3, 300.0), (4e3, 77.0))
    attendue = 4 * BOLTZMANN * 1e3 * 4e3 * (300 * 4e3 + 77 * 1e3) / 5e3**2
    assert densite == pytest.approx(attendue, rel=1e-9, abs=0)
    superposition = (
        densite_thermique(1e3, 300.0) * (4e3 / 5e3) ** 2
        + densite_thermique(4e3, 77.0) * (1e3 / 5e3) ** 2
    )
    assert densite == pytest.approx(superposition, rel=1e-9, abs=0)
    isotherme = densite_de_deux_impedances_en_parallele((1e3, 300.0), (4e3, 300.0))
    assert isotherme == pytest.approx(densite_thermique(800.0, 300.0), rel=1e-9, abs=0)


def test_des_impedances_en_parallele_a_meme_temperature() -> None:
    """Planche 16, poussee : a `T` egale, `4 k_B T Re[Z_1 // Z_2]`, meme complexes."""
    z1, z2 = 1e3 + 2e3j, 4e3 - 1e3j
    densite = densite_de_deux_impedances_en_parallele((z1, 300.0), (z2, 300.0))
    parallele = z1 * z2 / (z1 + z2)
    assert densite == pytest.approx(
        densite_thermique(parallele, 300.0), rel=1e-9, abs=0
    )


def test_les_autres_bruits() -> None:
    """Planches 19 a 24 : `A I^2 / f`, `racine(2 q I B)`, `1 / f^n` avec `n ~ 2`."""
    rose = densite_spectrale_en_un_sur_f(2e-12 * 1e-3**2, 10.0)
    assert rose == pytest.approx(2e-19, rel=1e-9, abs=0)
    grenaille = math.sqrt(densite_spectrale_de_grenaille(1e-3) * 1e4)
    assert grenaille == pytest.approx(
        math.sqrt(2 * 1.602e-19 * 1e-3 * 1e4), rel=1e-3, abs=0
    )
    popcorn = densite_spectrale_en_un_sur_f(1.0, 10.0, exposant=2.0)
    assert popcorn == pytest.approx(0.01, rel=1e-9, abs=0)
    assert est_blanc("thermique")
    assert est_blanc("grenaille")
    assert not est_blanc("scintillement")


def test_la_frequence_de_coin() -> None:
    """Planches 30 et 31 : `P_w = K / f_c` ; `P_w (f_h - f_l + f_c ln(f_h / f_l))`."""
    modele = ModeleBlancEtRose(blanc=25.0, coin=15.0)
    assert modele.coefficient_rose() == 375.0
    assert modele.densite(15.0) == pytest.approx(2 * 25.0, rel=1e-9, abs=0)
    puissance = modele.puissance(0.1, 10.0)
    assert puissance == pytest.approx(
        25.0 * (9.9 + 15.0 * math.log(100)), rel=1e-9, abs=0
    )
    assert puissance == pytest.approx(
        puissance_integree_blanc_et_rose(25.0, 375.0, 0.1, 10.0), rel=1e-9, abs=0
    )
    plus_bas = ModeleBlancEtRose(blanc=25.0, coin=1.0)
    assert plus_bas.puissance(0.1, 10.0) < puissance
    with pytest.raises(ValueError, match="frequence de coin negative"):
        ModeleBlancEtRose(blanc=25.0, coin=-1.0)


OPA140 = {10.0: 8.0, 100.0: 5.8, 1000.0: 5.1}
"""Planche 33 : `e_n` en nV/racine(Hz), a trois frequences."""


def test_la_fiche_de_l_opa140_se_lit_comme_un_bruit_blanc_plus_rose() -> None:
    """Planches 33 et 34 : deux points fixent le modele, les autres le jugent.

    Ajuste sur 10 Hz et 1 kHz, le modele rend 5,4 a 100 Hz pour 5,8, et
    45 nV efficaces de 0,1 a 10 Hz pour 42 : « the formula may provide
    inaccurate results », dit la planche 32.
    """
    modele = ModeleBlancEtRose.depuis_deux_points((10.0, 8.0), (1000.0, 5.1))
    assert math.sqrt(modele.densite(10.0)) == pytest.approx(8.0, rel=1e-9, abs=0)
    assert math.sqrt(modele.densite(1000.0)) == pytest.approx(5.1, rel=1e-9, abs=0)
    assert modele.coin == pytest.approx(15.0, rel=1e-9, abs=0.1)
    assert math.sqrt(modele.densite(100.0)) == pytest.approx(
        OPA140[100.0], rel=0.07, abs=0
    )
    efficace = math.sqrt(modele.puissance(0.1, 10.0))
    assert efficace == pytest.approx(42.0, rel=0.08, abs=0)
    assert excursion_crete_a_crete(42.0) == pytest.approx(250.0, rel=0.11, abs=0)


def test_le_bruit_ramene_a_l_entree() -> None:
    """Planches 40 et 41 : `4 k_B T R_S + S_Vn + R_S^2 S_In`, puis sa racine."""
    tension, courant, source = (5e-9) ** 2, (2e-12) ** 2, 10e3
    densite = densite_ramenee_a_l_entree(source, tension, courant, temperature=290.0)
    thermique = densite_thermique(source, 290.0)
    assert densite == pytest.approx(
        thermique + tension + source**2 * courant, rel=1e-9, abs=0
    )
    intrinseque = tension_efficace_ramenee_a_l_entree(5e-9, 2e-12, source)
    assert densite - thermique == pytest.approx(intrinseque**2, rel=1e-9, abs=0)


def test_la_tension_domine_a_basse_resistance_de_source() -> None:
    """Planche 42 : le bruit en courant ne compte que par `R_S I_n`."""
    assert bruit_dominant_selon_la_source(5e-9, 2e-12, 100.0) == "tension"
    assert bruit_dominant_selon_la_source(5e-9, 2e-12, 1e6) == "courant"


def test_le_facteur_de_bruit_du_modele() -> None:
    """Planches 48 a 55 : `F = 1 + (V_n^2 + R_S^2 I_n^2) / (4 k_B T R_S B)`."""
    tension, courant, bande = 50e-9, 20e-12, 1e3
    for source in (500.0, 2500.0, 50e3):
        facteur = facteur_de_bruit_du_modele(tension, courant, source, bande=bande)
        thermique = tension_efficace_thermique(source, 290.0, bande) ** 2
        intrinseque = tension**2 + (source * courant) ** 2
        assert facteur == pytest.approx(1 + intrinseque / thermique, rel=1e-9, abs=0)
        assert facteur >= 1
        assert figure_de_bruit(facteur) == pytest.approx(
            10 * math.log10(facteur), rel=1e-9, abs=0
        )
    sans_courant = facteur_de_bruit_du_modele(tension, 0.0, 500.0, bande=bande)
    assert sans_courant - 1 == pytest.approx(
        tension**2 / tension_efficace_thermique(500.0, 290.0, bande) ** 2,
        rel=1e-9,
        abs=0,
    )


def test_la_resistance_de_source_qui_minimise_le_facteur_de_bruit() -> None:
    """Planche 55 : `R_S = V_n / I_n` ; mais le bruit minimal reste a `R_S = 0`."""
    tension, courant, bande = 50e-9, 20e-12, 1e3
    optimale = resistance_de_source_optimale(tension, courant)
    assert optimale == pytest.approx(2500.0, rel=1e-9, abs=0)
    au_minimum = facteur_de_bruit_du_modele(tension, courant, optimale, bande=bande)
    for voisine in (0.8 * optimale, 1.25 * optimale):
        ailleurs = facteur_de_bruit_du_modele(tension, courant, voisine, bande=bande)
        assert ailleurs > au_minimum

    def puissance(source: float) -> float:
        densite = densite_ramenee_a_l_entree(
            source, tension**2 / bande, courant**2 / bande, temperature=290.0
        )
        return densite * bande

    assert puissance(0.0) == pytest.approx(tension**2, rel=1e-9, abs=0)
    assert puissance(0.0) < puissance(optimale)
    with pytest.raises(ValueError, match="bruit en courant nul"):
        resistance_de_source_optimale(tension, 0.0)


def test_la_definition_par_les_rapports_signal_sur_bruit() -> None:
    """Planches 58 a 60 : `F = SNR_in / SNR_out` redonne la premiere definition."""
    tension, courant, source, bande, gain = 50e-9, 20e-12, 2500.0, 1e3, 40.0
    signal = 1e-6**2
    thermique = tension_efficace_thermique(source, 290.0, bande) ** 2
    intrinseque = tension**2 + (source * courant) ** 2
    entree = signal / thermique
    sortie = gain**2 * signal / (gain**2 * (thermique + intrinseque))
    attendu = facteur_de_bruit_du_modele(tension, courant, source, bande=bande)
    assert facteur_de_bruit(entree, sortie) == pytest.approx(attendu, rel=1e-9, abs=0)


def test_un_filtre_etroit_ferait_descendre_le_facteur_sous_un() -> None:
    """Planche 61 : le defaut de la definition par les rapports, hors gain constant."""
    signal, bruit_d_entree = 1.0, 0.5
    bruit_garde = bruit_d_entree / 100 + 1e-3
    with pytest.raises(ValueError, match="facteur de bruit sous l'unite"):
        facteur_de_bruit(signal / bruit_d_entree, signal / bruit_garde)


def test_le_facteur_de_bruit_ponctuel() -> None:
    """Planches 62 a 64 : `F(f) = 1 + (S_Vn + R_S^2 S_In) / (4 k_B T R_S)`."""
    modele = ModeleBlancEtRose(blanc=(5e-9) ** 2, coin=15.0)
    source = 10e3
    thermique = densite_thermique(source, 290.0)
    facteurs = []
    for frequence in (1.0, 15.0, 1e3):
        totale = densite_ramenee_a_l_entree(
            source, modele.densite(frequence), 0.0, temperature=290.0
        )
        facteur = facteur_de_bruit_ponctuel(totale, thermique)
        assert facteur == pytest.approx(
            1 + modele.densite(frequence) / thermique, rel=1e-9, abs=0
        )
        facteurs.append(facteur)
    assert facteurs[0] > facteurs[1] > facteurs[2] > 1


def test_les_gardes_des_modeles_de_bruit() -> None:
    """Temperature, densites, bandes et frequences negatives sont refusees."""
    with pytest.raises(ValueError, match="temperature absolue negative"):
        densite_thermique(1e3, -1.0)
    with pytest.raises(ValueError, match="impedances opposees"):
        densite_de_deux_impedances_en_parallele((1e3j, 300.0), (-1e3j, 300.0))
    with pytest.raises(ValueError, match="densite blanche negative"):
        ModeleBlancEtRose(blanc=-1.0, coin=15.0)
    with pytest.raises(ValueError, match="frequences invalides ou confondues"):
        ModeleBlancEtRose.depuis_deux_points((10.0, 8.0), (10.0, 5.1))
    modele = ModeleBlancEtRose(blanc=25.0, coin=15.0)
    with pytest.raises(ValueError, match="frequence nulle ou negative"):
        modele.densite(0.0)
    with pytest.raises(ValueError, match="bande invalide"):
        modele.puissance(10.0, 0.1)
    with pytest.raises(ValueError, match="densite negative"):
        densite_ramenee_a_l_entree(1e3, -1.0, 0.0)
    with pytest.raises(ValueError, match="source sans bruit thermique"):
        facteur_de_bruit_du_modele(50e-9, 20e-12, 0.0, bande=1e3)
