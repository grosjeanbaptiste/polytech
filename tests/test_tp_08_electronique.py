"""TP 8 d'electronique : le bruit thermique des resistances d'un montage a AOP.

Oracles : le corrige officiel (`ElectroniqueInstrumentation/TP/
TP_8_Solution.pdf`, trois pages) -- 4,06 nV par racine de hertz pour
1 kohm a 298 K, 40 pour 100 kohm, 0,1616 (uV)^2/Hz et 402 nV en sortie,
3,98 et 4,02 nV rapportes a l'entree.

La bibliotheque ne reprend pas ses gains : chaque source de bruit est une
source de tension en serie avec sa resistance, et l'analyse nodale du
montage, AOP de gain fini passe a la limite, rend son transfert vers la
sortie. La regle `S_YY = |H|^2 S_XX` vient de `polytech.signaux`.

Un ecart, d'arrondi mais qui se propage : l'officiel remplace 4,06 par 4
des la premiere question. Sans cet arrondi, la sortie vaut 407,7 nV et
non 402, et les deux densites rapportees a l'entree 4,04 et 4,08.
"""

from __future__ import annotations

import math

import pytest

from polytech.electronique.application import bruit_tp_08 as br

K = 1.38e-23
"""La constante de Boltzmann telle que l'officiel l'arrondit."""
K_EXACTE = 1.380649e-23
"""Sa valeur exacte, fixee par la definition du kelvin (SI 2019)."""


def test_ex21_la_densite_d_un_kilohm() -> None:
    """Page 1 : `racine(4 k_B R T) = 4,06 nV / racine(Hz)` a 298 K."""
    assert br.TEMPERATURE == 298.0
    densite = br.densite_efficace(1e3)
    assert densite * 1e9 == pytest.approx(4.06, abs=5e-3)
    assert densite == pytest.approx(math.sqrt(4 * K * 1e3 * 298), rel=1e-3, abs=0)
    assert densite**2 == pytest.approx(1.645e-17, rel=1e-3, abs=0)


def test_ex21_cent_fois_la_resistance_dix_fois_le_bruit() -> None:
    """Page 1 : 40 nV depuis les 4 nV de l'enonce ; 40,57 sans l'arrondi."""
    assert br.depuis_la_valeur_de_l_enonce(100e3) * 1e9 == pytest.approx(40.0)
    assert br.densite_efficace(100e3) / br.densite_efficace(1e3) == pytest.approx(10.0)
    exacte = math.sqrt(4 * K_EXACTE * 298 * 100e3)
    assert br.densite_efficace(100e3) == pytest.approx(exacte)
    assert exacte * 1e9 == pytest.approx(40.57, abs=5e-3)
    with pytest.raises(ValueError, match="resistance nulle ou negative"):
        br.depuis_la_valeur_de_l_enonce(0.0)


def test_ex22_les_transferts_des_deux_sources_de_bruit() -> None:
    """Page 2 : `-R_2 / R_1 = -100` pour le bruit de `R_1`, 1 pour celui de `R_2`."""
    de_r1, de_r2 = br.transferts_des_bruits()
    assert de_r1 == -100
    assert abs(de_r2) == 1


def test_ex22_les_gains_des_deux_montages() -> None:
    """Page 3 : 101 pour le non-inverseur, -100 pour l'inverseur."""
    assert br.gain_du_signal(br.Montage.NON_INVERSEUR) == 101
    assert br.gain_du_signal(br.Montage.INVERSEUR) == -100


def test_ex22_la_densite_en_sortie() -> None:
    """Page 2 : `4 k_B T (10^4 R_1 + R_2)`, la meme pour les deux montages."""
    sortie = br.densite_en_sortie()
    attendue = 4 * K * 298 * (1e4 * 1e3 + 100e3)
    assert sortie == pytest.approx(attendue, rel=1e-3, abs=0)
    exacte = 4 * K_EXACTE * 298 * (1e4 * 1e3 + 100e3)
    assert sortie == pytest.approx(exacte, rel=1e-9, abs=0)
    assert math.sqrt(exacte) * 1e9 == pytest.approx(407.7, abs=0.05)


def test_ex22_les_nombres_de_l_officiel_viennent_de_quatre_nanovolts() -> None:
    """Page 2 : 0,1616 (uV)^2/Hz et 402 nV supposent `4,06 = 4`."""
    arrondie = br.densite_en_sortie(par_kilohm=4e-9)
    assert arrondie * 1e12 == pytest.approx(0.1616, abs=5e-5)
    assert math.sqrt(arrondie) * 1e9 == pytest.approx(402.0, abs=0.05)
    assert math.sqrt(br.densite_en_sortie()) * 1e9 - 402.0 > 5.0
    with pytest.raises(ValueError, match="densite nulle ou negative"):
        br.densite_en_sortie(par_kilohm=0.0)


def test_ex22_la_part_de_chaque_resistance() -> None:
    """`10^4 R_1` contre `R_2` : 99 % du bruit de sortie vient de `R_1`."""
    de_r1, de_r2 = br.parts_du_bruit_de_sortie()
    assert de_r1 == pytest.approx(10_000 / 10_100)
    assert de_r1 + de_r2 == pytest.approx(1.0)


def test_ex22_la_densite_rapportee_a_l_entree() -> None:
    """Page 3 : 3,98 et 4,02 nV avec l'arrondi ; 4,04 et 4,08 sans."""
    gauche = br.rapportee_a_l_entree(br.Montage.NON_INVERSEUR)
    droite = br.rapportee_a_l_entree(br.Montage.INVERSEUR)
    sortie = math.sqrt(4 * K_EXACTE * 298 * (1e4 * 1e3 + 100e3))
    assert (gauche, droite) == pytest.approx(
        (sortie / 101, sortie / 100), rel=1e-9, abs=0
    )
    assert (gauche * 1e9, droite * 1e9) == pytest.approx((4.04, 4.08), abs=5e-3)
    arrondies = (
        br.rapportee_a_l_entree(br.Montage.NON_INVERSEUR, par_kilohm=4e-9),
        br.rapportee_a_l_entree(br.Montage.INVERSEUR, par_kilohm=4e-9),
    )
    assert [a * 1e9 for a in arrondies] == pytest.approx([3.98, 4.02], abs=5e-3)


def test_ex22_la_resistance_qui_ferait_le_meme_bruit() -> None:
    """A l'entree : `R_1 // R_2` pour le non-inverseur, `R_1 (1 + R_1/R_2)` sinon."""
    gauche = br.resistance_equivalente(br.Montage.NON_INVERSEUR)
    droite = br.resistance_equivalente(br.Montage.INVERSEUR)
    assert gauche == pytest.approx(1e3 * 100e3 / 101e3)
    assert droite == pytest.approx(1e3 * (1 + 1e3 / 100e3))
    assert gauche < 1e3 < droite


def test_ex22_le_gain_de_bruit() -> None:
    """La sortie est le bruit de `R_1 // R_2` amplifie par `1 + R_2/R_1`."""
    parallele = br.densite_efficace(1e3 * 100e3 / 101e3)
    assert 101 * parallele == pytest.approx(math.sqrt(br.densite_en_sortie()))


def test_une_mise_a_l_echelle_des_impedances_change_le_bruit() -> None:
    """Resistances divisees par 100 : gains inchanges, bruit divise par 10."""
    reference = math.sqrt(br.densite_en_sortie())
    reduite = math.sqrt(br.densite_en_sortie(echelle=0.01))
    assert reduite == pytest.approx(reference / 10)
    with pytest.raises(ValueError, match="facteur d'echelle nul ou negatif"):
        br.densite_en_sortie(echelle=0.0)
