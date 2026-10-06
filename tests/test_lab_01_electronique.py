"""Laboratoire 1 d'electronique : filtrage -- RC passe-haut, RLC passe-bas, Bode.

Oracles : le corrige officiel, manuscrit
(`ElectroniqueInstrumentation/LABOS/Lab_01_Filtrage_Solution.pdf`, quinze
pages lues sur le scan) -- ses formules, ses quatre modules a la resonance
(1,79 et 0,27 avec le generateur, 2,86 et 0,29 sans), sa frequence de
resonance (55,6 kHz) et ses releves (28 dB, 8 dB, -10,5 dB).

La bibliotheque ne recopie pas ces formules : chaque fonction de transfert
est le diviseur d'impedances du montage, et les modules sont evalues sur
elle.

Trois ecarts, relus sur le scan :

* page 3, question 3.2 : « H ~ 1/(j w C) » et « H ~ j w L » sont les
  impedances dominantes, non la fonction de transfert, qui tend vers 1
  puis vers `-1 / (w^2 L C)` ;
* page 15 : `pi / 6` est pris pour 60 degres ; il vaut 30 ;
* page 15 : -10 dB lus a 1000 rad/s, quand la droite qui passe par les
  deux autres lectures (40 dB a 100, -40 dB a 10^4) y vaut 0 dB.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.application import bode_lab_01 as bo
from polytech.electronique.application import courbes_lab_01 as co
from polytech.electronique.application import rc_lab_01 as rc
from polytech.electronique.application import rlc_lab_01 as rl

P = sp.Symbol("p")


# --- 2.3 : le RC passe-haut -------------------------------------------


def test_le_passe_haut_est_un_diviseur_d_impedances() -> None:
    """Page 1 : `H(p) = p R C / (1 + p R C)`, zero a l'origine, pole en `-1/RC`."""
    r, c = sp.symbols("R C", positive=True)
    transfert = rc.passe_haut(r, c, P)
    assert sp.simplify(transfert - P * r * c / (1 + P * r * c)) == 0
    assert sp.solve(sp.numer(sp.together(transfert)), P) == [0]
    assert sp.solve(sp.denom(sp.together(transfert)), P) == [-1 / (r * c)]


def test_a_la_coupure_le_module_et_la_phase() -> None:
    """Pages 1 et 2 : `1 / racine(2)` et +45 degres en `w = 1 / RC`."""
    module, phase = rc.reponse_a_la_coupure()
    assert module == pytest.approx(1 / math.sqrt(2))
    assert phase == pytest.approx(45.0)
    assert rc.pulsation_de_coupure() == pytest.approx(1e4)
    assert rc.frequence_de_coupure() == pytest.approx(1591.5, abs=0.05)


def test_les_deux_asymptotes_du_passe_haut() -> None:
    """Question 1 : `j w R C` aux basses frequences, 1 aux hautes."""
    basse, haute = rc.asymptotes(P)
    r, c = rc.RESISTANCE, rc.CAPACITE
    assert sp.simplify(basse - P * sp.nsimplify(r * c)) == 0
    assert haute == 1


def test_question_2_le_generateur_pese_six_pour_cent() -> None:
    """Page 2 : `R_G / R = 600 / 10 000 = 6 %`."""
    assert rc.poids_du_generateur() == pytest.approx(0.06)
    plateau, coupure = rc.effet_du_generateur()
    assert plateau == pytest.approx(10_000 / 10_600)
    assert 20 * math.log10(plateau) == pytest.approx(-0.506, abs=5e-4)
    assert coupure == pytest.approx(1 / (10_600 * 10e-9) / (2 * math.pi))
    assert coupure == pytest.approx(1501.5, abs=0.05)


# --- 2.4 : le RLC passe-bas -------------------------------------------


def test_le_rlc_est_un_diviseur_d_impedances() -> None:
    """Page 3 : `H(p) = 1 / (p^2 L C + p R C + 1)`."""
    r, ind, c = sp.symbols("R L C", positive=True)
    transfert = rl.passe_bas(r, ind, c, P)
    assert sp.simplify(transfert - 1 / (P**2 * ind * c + P * r * c + 1)) == 0


def test_question_3_les_asymptotes_du_rlc() -> None:
    """Le transfert tend vers 1, puis vers `1 / (p^2 L C)` : jamais une impedance."""
    basse, haute = rl.asymptotes(P)
    produit = sp.nsimplify(rl.INDUCTANCE * rl.CAPACITE)
    assert basse == 1
    assert sp.simplify(haute - 1 / (P**2 * produit)) == 0
    a_cent_fois = abs(complex(haute.subs(P, sp.I * 100 * rl.pulsation_de_resonance())))
    assert a_cent_fois == pytest.approx(1e-4)


def test_question_4_la_resonance() -> None:
    """Page 4 et 7 : `w_R = 1 / racine(L C)`, soit `3,5 10^5 rad/s` et 55,6 kHz."""
    assert rl.pulsation_de_resonance() == pytest.approx(3.5e5, rel=5e-3)
    assert rl.pulsation_de_resonance() == pytest.approx(1 / math.sqrt(8.2e-3 * 1e-9))
    assert rl.frequence_de_resonance() == pytest.approx(55.6e3, rel=1e-3)
    assert rl.impedance_caracteristique() == pytest.approx(math.sqrt(8.2e6))


@pytest.mark.parametrize(
    ("resistance", "serie", "attendu"),
    [(1e3, 600.0, 1.79), (10e3, 600.0, 0.27), (1e3, 0.0, 2.86), (10e3, 0.0, 0.29)],
)
def test_question_4_le_module_a_la_resonance(
    resistance: float, serie: float, attendu: float
) -> None:
    """Page 4 : `racine(L / C) / R`, avec ou sans les 600 ohm du generateur."""
    module = rl.module_a_la_resonance(resistance, generateur=serie > 0)
    assert module == pytest.approx(attendu, abs=6e-3)
    assert module == pytest.approx(math.sqrt(8.2e-3 / 1e-9) / (resistance + serie))


def test_a_la_resonance_le_transfert_est_imaginaire_pur() -> None:
    """Page 4 : `H(j w_R) = -j racine(L / C) / R`."""
    reponse = rl.reponse_a_la_resonance(1e3)
    assert reponse.real == pytest.approx(0.0, abs=1e-9)
    assert reponse.imag == pytest.approx(-math.sqrt(8.2e6) / 1e3)


def test_la_nature_des_poles() -> None:
    """Page 3, annotations : complexes pour 1 kohm, reels pour 10 kohm."""
    assert rl.resistance_critique() == pytest.approx(2 * math.sqrt(8.2e6))
    assert rl.resistance_critique() == pytest.approx(5727.0, abs=0.5)
    assert not rl.poles_reels(1e3 + 600)
    assert rl.poles_reels(10e3 + 600)
    assert rl.poles_reels(10e3)
    assert not rl.poles_reels(0.0)
    lent, rapide = rl.poles(10e3)
    assert lent * rapide == pytest.approx(rl.pulsation_de_resonance() ** 2)
    assert lent + rapide == pytest.approx(-10e3 / 8.2e-3)
    assert lent > rapide
    with pytest.raises(ValueError, match="poles complexes"):
        rl.poles(1e3)


def test_question_5_les_tensions_aux_bornes_de_l_et_de_c() -> None:
    """Page 10 : `V_L + V_C = 0` a la resonance, chacune valant `Q` fois `V_1`."""
    sur_c, sur_l, sur_les_deux = rl.tensions_a_la_resonance(1e3)
    assert abs(sur_c) == pytest.approx(2.86, abs=6e-3)
    assert abs(sur_l) == pytest.approx(abs(sur_c))
    assert abs(sur_les_deux) == pytest.approx(0.0, abs=1e-9)
    assert abs(sur_c) + abs(sur_l) > 5


def test_la_resistance_que_les_releves_revelent() -> None:
    """Page 8 : 28 dB sans resistance, 8 dB avec 1 kohm, la ou la theorie dit plus."""
    sans = rl.resistance_parasite(28.0, 0.0)
    avec = rl.resistance_parasite(8.0, 1e3)
    assert sans == pytest.approx(math.sqrt(8.2e6) / 10 ** (28 / 20))
    assert sans == pytest.approx(114.0, abs=0.5)
    assert avec == pytest.approx(140.0, abs=0.5)
    assert rl.resistance_parasite(-10.5, 10e3) < 0
    with pytest.raises(ValueError, match="resistance negative"):
        rl.resistance_parasite(8.0, -1.0)


def test_question_6_la_tension_mesuree_s_effondre_a_la_resonance() -> None:
    """Page 12 : `V_1 / V_in = R / (R + R_G)` a la resonance."""
    assert rl.tension_mesuree_a_la_resonance(1e3) == pytest.approx(1000 / 1600)
    assert rl.tension_mesuree_a_la_resonance(10e3) == pytest.approx(10_000 / 10_600)
    assert rl.tension_mesuree_a_la_resonance(0.0) == 0.0
    avec_bobine = rl.tension_mesuree_a_la_resonance(0.0, parasite=114.0)
    assert avec_bobine == pytest.approx(114 / 714)
    assert rl.tension_mesuree(1e3, 100.0) == pytest.approx(1.0, abs=1e-3)
    a_la_resonance = rl.tension_mesuree(1e3, rl.frequence_de_resonance())
    assert a_la_resonance == pytest.approx(1000 / 1600)


# --- 2.5 : les exercices de Bode --------------------------------------


def test_question_7_premiere_fonction() -> None:
    """Page 13 : -60 dB, puis +20 dB par decade de 10^3 a 10^6, puis 0 dB."""
    points = bo.asymptote_de_la_premiere()
    assert points == pytest.approx([(1e3, -60.0), (1e6, 0.0)])
    assert bo.gain_exact(bo.PREMIERE, 1e3) == pytest.approx(-60 + 3.01, abs=0.01)
    assert bo.phases_en_marches(bo.PREMIERE) == (0.0, 90.0, 0.0)
    assert bo.phase_exacte(bo.PREMIERE, math.sqrt(1e9)) == pytest.approx(86.4, abs=0.05)


def test_question_7_seconde_fonction() -> None:
    """Page 14 : -20 dB par decade, creux a -40 dB en 10^3, puis 20 dB en 10^6."""
    points = bo.asymptote_de_la_seconde()
    assert points == pytest.approx([(1.0, 20.0), (1e3, -40.0), (1e6, 20.0)])
    assert bo.gain_exact(bo.SECONDE, 1e3) == pytest.approx(-40 + 6.02, abs=0.01)
    assert bo.phases_en_marches(bo.SECONDE) == (-90.0, 90.0, 0.0)
    assert bo.gain_exact(bo.SECONDE, 1e9) == pytest.approx(20.0, abs=1e-3)


def test_question_8_les_trois_sorties() -> None:
    """Page 15 : 100 V et -90 degres ; 50 mV ; puis 2 V et non 640 mV."""
    premiere, deuxieme, troisieme = bo.sorties()
    assert premiere == pytest.approx((100.0, -90.0))
    assert deuxieme[0] == pytest.approx(0.05)
    assert troisieme[0] == pytest.approx(2.0)
    assert troisieme[1] == pytest.approx(30.0 + bo.LECTURES[1e3][1])
    assert bo.sortie_de_l_officiel()[0] == pytest.approx(0.64, abs=0.01)
    assert bo.sortie_de_l_officiel()[1] == pytest.approx(-110.0)


def test_question_8_la_lecture_a_mille_radians_par_seconde() -> None:
    """La droite par (100 ; 40 dB) et (10^4 ; -40 dB) passe par 0 dB en 10^3."""
    assert bo.gain_interpole(1e3) == pytest.approx(0.0, abs=1e-9)
    assert bo.pente_entre_les_lectures() == pytest.approx(-40.0)
    assert bo.LECTURES[1e3][0] == 0.0
    assert math.degrees(math.pi / 6) == pytest.approx(30.0)


# --- Les courbes du document ------------------------------------------


def test_la_courbe_du_passe_haut() -> None:
    """De 10 Hz a 100 kHz, la plage du releve ; -3 dB a la coupure."""
    frequences, gains = co.passe_haut()
    assert (frequences[0], frequences[-1]) == pytest.approx((10.0, 1e5))
    assert gains[-1] == pytest.approx(0.0, abs=0.01)
    assert gains[0] == pytest.approx(
        20 * math.log10(10 / rc.frequence_de_coupure()), abs=0.01
    )
    asymptote = co.asymptote_du_passe_haut()
    assert [f for f, _ in asymptote] == pytest.approx(
        [10.0, rc.frequence_de_coupure(), 1e5]
    )
    assert [g for _, g in asymptote] == pytest.approx([gains[0], 0.0, 0.0], abs=0.01)


@pytest.mark.parametrize("resistance", [114.0, 1e3, 10e3])
def test_les_courbes_du_rlc(resistance: float) -> None:
    """Le module de `V_C / V_1`, de 100 Hz a 100 kHz, la plage des releves."""
    frequences, gains = co.passe_bas(resistance)
    assert (frequences[0], frequences[-1]) == pytest.approx((100.0, 1e5))
    assert gains[0] == pytest.approx(0.0, abs=0.01)
    attendu = 20 * math.log10(rl.impedance_caracteristique() / resistance)
    proche = min(range(len(frequences)), key=lambda i: abs(frequences[i] - 55.58e3))
    assert gains[proche] == pytest.approx(attendu, abs=1e-6)


def test_les_courbes_de_la_question_7() -> None:
    """Les deux fonctions, de 1 a `10^8 rad/s`, comme sur le corrige."""
    for transfert in (bo.PREMIERE, bo.SECONDE):
        pulsations, gains = co.bode(transfert)
        assert (pulsations[0], pulsations[-1]) == pytest.approx((1.0, 1e8))
        assert gains[-1] == pytest.approx(bo.gain_exact(transfert, 1e8), abs=1e-6)
    _, gains = co.bode(bo.SECONDE)
    assert min(gains) == pytest.approx(-33.98, abs=0.05)


def test_les_lignes_brisees_de_la_question_7() -> None:
    """Chaque asymptote va d'un bord a l'autre de la figure, cassures comprises."""
    premiere, seconde = co.asymptotes_de_la_question_7()
    assert [w for w, _ in premiere] == [1.0, 1e3, 1e6, 1e8]
    assert [g for _, g in premiere] == pytest.approx([-60.0, -60.0, 0.0, 0.0])
    assert [w for w, _ in seconde] == [1.0, 1e3, 1e6, 1e8]
    assert [g for _, g in seconde] == pytest.approx([20.0, -40.0, 20.0, 20.0])
