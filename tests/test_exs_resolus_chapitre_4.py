"""Signaux et systemes : les exercices resolus, chapitre 4 (0.22 a 0.28).

Source : `SignauxSystemes/EXS/Exs_01_02_04_05_Resolus.pdf`, pages 9 a 20.
Les solutions -- manuscrites pour 0.22 et 0.23, imprimees pour les
autres, illustrees de traces MATLAB -- sont recopiees comme oracles et
recoupees ; aucune n'est contredite. Pour 0.26, le professeur a confirme le
28 octobre 2026 que Mason donne `p/(p+1)`.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.maths.domain.comparaison import egales, proches, proches_relativement
from polytech.signaux.application import resolus_graphes as gr
from polytech.signaux.application import resolus_quadripoles as qu
from polytech.signaux.domain.symboles import IMAGE as P

# --- 0.22 : quatre quadripoles du premier ordre -----------------------------


def test_0_22_les_quatre_transmittances() -> None:
    """Pages 12 et 13 : celles des appels MATLAB `[1],[RC 1]`, `[R],[L R]`, etc."""
    tau_rc, tau_lr = sp.Rational(1, 10**7), sp.Rational(1, 10**5)
    attendues = {
        "RC": 1 / (tau_rc * P + 1),
        "LR": 1 / (tau_lr * P + 1),
        "CR": tau_rc * P / (tau_rc * P + 1),
        "RL": tau_lr * P / (tau_lr * P + 1),
    }
    for nom, attendue in attendues.items():
        assert egales(qu.transmittance(nom), attendue)


def test_0_22_les_limites_encadrees_du_manuscrit() -> None:
    """Pages 12 et 13 : `h` puis `a`, en zero puis a l'infini."""
    assert qu.limites("RC") == {"h": (10**7, 0), "a": (0, 1)}
    assert qu.limites("RL") == {"h": (sp.oo, 0), "a": (1, 0)}


def test_0_22_les_hauteurs_des_traces_matlab() -> None:
    """Page 14 : les impulsionnelles partent de `+-10^7` et de `+-10^5`."""
    assert qu.depart_de_l_impulsionnelle("RC") == 10**7
    assert qu.depart_de_l_impulsionnelle("LR") == 10**5
    assert qu.depart_de_l_impulsionnelle("CR") == -(10**7)
    assert qu.depart_de_l_impulsionnelle("RL") == -(10**5)


def test_0_22_les_passe_haut_portent_une_impulsion() -> None:
    """Page 14 : « les impulsions de Dirac [...] ne sont pas portees en graphique »."""
    assert not qu.porte_une_impulsion("RC")
    assert not qu.porte_une_impulsion("LR")
    assert qu.porte_une_impulsion("CR")
    assert qu.porte_une_impulsion("RL")


# --- 0.23 et 0.24 : appariement, puis le circuit de l'exemple 4.3 -----------


def test_0_23_l_appariement_de_la_solution() -> None:
    """Page 15 : `C-a-ii, B-b-v, E-c-i, D-d-iv, A-e-iii`."""
    assert qu.appariement() == {
        "A": ("e", 3),
        "B": ("b", 5),
        "C": ("a", 2),
        "D": ("d", 4),
        "E": ("c", 1),
    }


def test_0_24_la_transmittance_de_la_solution() -> None:
    """Page 16 : `R / (p^2 RLC + p (R^2 C + L) + 2R)`."""
    r, ind, cap = qu.R, qu.L, qu.C
    attendue = r / (P**2 * r * ind * cap + P * (r**2 * cap + ind) + 2 * r)
    assert egales(qu.transmittance_de_l_exemple(), attendue)


def test_0_24_les_quatre_limites_de_la_solution() -> None:
    """Page 17 : `h` va de 0 a 0 ; `a` de 0 a 1/2."""
    assert qu.limites_de_l_exemple() == {"h": (0, 0), "a": (0, sp.Rational(1, 2))}


def test_0_24_les_reperes_des_traces_matlab() -> None:
    """Page 16 : l'impulsionnelle culmine vers 320 a 0,8 ms, l'indicielle vers 0,52."""
    instant, sommet = qu.sommet_de_l_impulsionnelle()
    assert proches_relativement(instant, math.pi / 4 * 1e-3, 1e-6)
    assert proches(sommet, 322, 1)
    assert proches(qu.sommet_de_l_indicielle(), 0.52, 5e-3)


# --- 0.25 a 0.28 : des graphes ----------------------------------------------


def test_0_25_la_transmittance_de_la_solution() -> None:
    """Page 17 : `p / (p^2 + 20p + 200)` ; pulsation 10 rad/s, constante 1/10 s."""
    assert egales(gr.transmittance("0.25"), P / (P**2 + 20 * P + 200))
    assert gr.sigma_et_omega("0.25") == (10, 10)
    assert gr.limites("0.25") == {"h": (1, 0), "a": (0, 0)}


def test_0_25_le_sommet_du_trace_matlab() -> None:
    """Page 18 : l'indicielle culmine vers 0,032, a 0,08 s."""
    instant, sommet = gr.sommet_de_l_indicielle("0.25")
    assert proches_relativement(instant, math.pi / 40, 1e-6)
    assert proches(sommet, 0.0322, 5e-5)


def test_0_26_la_regle_de_mason_donne_p_sur_p_plus_un() -> None:
    """Page 18 : `p/(p+1)` ; la boucle `-1/p` enferme les deux branches."""
    assert egales(gr.transmittance("0.26"), P / (P + 1))


def test_0_26_les_deux_branches_en_parallele_valent_un_dans_la_boucle() -> None:
    """Les branches se somment en `1` : `H = 1 / (1 + 1/p)`."""
    assert egales(1 / (P + 1) + P / (P + 1), 1)
    assert egales(gr.transmittance("0.26"), 1 / (1 + 1 / P))


def test_0_26_la_valeur_finale_de_la_solution() -> None:
    """Page 18 : `a(infini) = H(0) = 0` et `a(0+) = H(oo) = 1`."""
    assert gr.limites("0.26")["a"] == (1, 0)


def test_0_27_la_transmittance_et_les_limites_de_la_solution() -> None:
    """Pages 18 et 19 : `(p+1)/(p^2+20p+20)` ; `h` de 1 a 0, `a` de 0 a 1/20."""
    assert egales(gr.transmittance("0.27"), (P + 1) / (P**2 + 20 * P + 20))
    assert gr.limites("0.27") == {"h": (1, 0), "a": (0, sp.Rational(1, 20))}


def test_0_27_les_constantes_de_temps() -> None:
    """Page 19 : « 1 s et 1/20 s » ; exactement 0,947 s et 0,053 s."""
    lente, rapide = gr.constantes_de_temps("0.27")
    assert proches(lente, 0.947, 5e-4)
    assert proches(rapide, 0.0528, 5e-5)


def test_0_27_le_zero_efface_presque_le_pole_lent() -> None:
    """Le zero en -1 est a 0,056 du pole lent : son mode ne pese que 0,3 %."""
    lent, rapide = gr.poids_des_modes_de_l_impulsionnelle("0.27")
    assert proches(lent, -0.0031, 5e-5)
    assert proches_relativement(lent + rapide, 1, 1e-6)


def test_0_28_la_transmittance_de_la_solution() -> None:
    """Page 19 : `5 / (p^3 + p^2 + 10p + 15)`."""
    attendue = 5 / (P**3 + P**2 + 10 * P + 15)
    assert egales(gr.transmittance("0.28"), attendue)


def test_0_28_le_systeme_est_instable() -> None:
    """Page 20 : des poles a droite, une oscillation a 3 rad/s (T = 2 s) qui diverge."""
    reel, pulsation = gr.pole_instable("0.28")
    assert proches(reel, 0.21, 5e-3)
    assert proches(pulsation, 3.25, 5e-3)
    assert proches(2 * math.pi / pulsation, 2, 0.1)


def test_0_28_la_valeur_finale_n_existe_pas() -> None:
    """Page 20 : le theoreme rendrait 1/3 ; la bibliotheque refuse de le rendre."""
    assert gr.limites("0.28")["a"][1] == sp.Rational(1, 3)
    with pytest.raises(ValueError, match="pole"):
        gr.valeur_finale_verifiee("0.28")
    assert gr.valeur_finale_verifiee("0.27") == sp.Rational(1, 20)
