"""Les planches d'electronique : l'amplificateur operationnel reel (2, II).

Source : `ElectroniqueInstrumentation/SLIDES/Slides_2_part2.pdf`, 85
planches lues sur leur couche de texte et, pour les schemas, sur les
pages rendues.

Les oracles : le gain de l'amplificateur a capacite de retour (planches
7 a 13), le systeme d'analyse nodale des planches 16 et 17, le CA3140
(planche 20 : 100 dB et 4,5 MHz, soit 45 Hz de bande ouverte), le gain
ferme a un pole (planches 22 a 26), la paire differentielle (planche 34), les
definitions des imperfections et les cinq montages de la fin.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electronique.domain.aop import aop, sortie_ideale
from polytech.electronique.domain.aop_interne import (
    TENSION_THERMIQUE,
    bande_en_boucle_ouverte,
    courant_de_decalage,
    courant_de_la_paire_differentielle,
    courant_de_polarisation,
    gain_ferme_a_un_pole,
    impedances_de_miller,
    sortie_avec_imperfections,
    tension_de_mode_commun,
    vitesse_de_balayage,
)
from polytech.electronique.domain.boucle_fermee import (
    bande_apres_retroaction,
    produit_gain_bande,
    vitesse_de_balayage_minimale,
)
from polytech.electronique.domain.defauts_aop import (
    capacite_de_miller,
    rejection_du_mode_commun,
    rejection_en_decibels,
)
from polytech.electronique.domain.montages import (
    convertisseur_courant_tension,
    derivateur,
    integrateur,
    somme_ponderee,
    soustracteur,
    transfert_a_impedances,
)
from polytech.electronique.domain.netliste import (
    Admittance,
    SourceDeTension,
    SourceDeTensionCommandee,
    resoudre,
)

S = sp.Symbol("s")
R, C, A = sp.symbols("R C A", positive=True)


def test_le_theoreme_de_miller() -> None:
    """Planche 5 : `Z_1 = Z / (1 - K)` et `Z_2 = Z / (1 - 1/K)`, ici `K = -A`."""
    entree, sortie = impedances_de_miller(1 / (S * C), -A)
    assert sp.simplify(entree - 1 / ((1 + A) * C * S)) == 0
    assert sp.simplify(sortie - 1 / ((1 + 1 / A) * C * S)) == 0
    assert sp.limit(sortie * S * C, A, sp.oo) == 1
    with pytest.raises(ValueError, match="gain unite"):
        impedances_de_miller(1 / (S * C), sp.Integer(1))


def test_l_effet_miller_sans_le_theoreme() -> None:
    """Planches 8 a 10 : `V_2 / V_i = -A / (sRC(1 + A) + 1)`, par l'analyse nodale."""
    elements = (
        SourceDeTension(1, 0, sp.Integer(1), sp.Symbol("i_e")),
        Admittance(1, 2, 1 / R),
        Admittance(2, 3, S * C),
        SourceDeTensionCommandee(3, 0, -A, (2, 0), sp.Symbol("i_a")),
    )
    sortie = resoudre(elements, 3)[2]
    assert sp.simplify(sortie + A / (S * R * C * (1 + A) + 1)) == 0
    assert sortie.subs(S, 0) == -A


def test_l_effet_miller_par_la_capacite_ramenee_a_l_entree() -> None:
    """Planches 10 et 11 : comme si `C_M = (1 + A) C` chargeait l'entree."""
    assert capacite_de_miller(12e-12, 10_000.0) == pytest.approx(12e-12 * 10_001)
    entree, _ = impedances_de_miller(1 / (S * C), -A)
    diviseur = entree / (entree + R)
    assert sp.simplify(-A * diviseur + A / (S * R * C * (1 + A) + 1)) == 0


R1, R2, R3, VS = sp.symbols("R_1 R_2 R_3 V_S", positive=True)


def test_la_masse_virtuelle_dans_l_analyse_nodale() -> None:
    """Planches 16 et 17 : six equations, dont `V_2 - V_3 = 0` ; `I_OA` en inconnue."""
    matrice = sp.Matrix(
        [
            [-1 / R1, 1 / R1, 0, 0, 1, 0],
            [1 / R1, -(1 / R1 + S * C), 0, 0, 0, 0],
            [0, 0, -(1 / R2 + 1 / R3), 1 / R2, 0, 0],
            [0, 0, 1 / R2, -1 / R2, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0],
        ]
    )
    membre = sp.Matrix([0, 0, 0, 0, VS, 0])
    planche = matrice.solve(membre)
    attendue = (1 + R2 / R3) * VS / (1 + S * R1 * C)
    assert sp.simplify(planche[3] - attendue) == 0
    assert sp.simplify(planche[1] - planche[2]) == 0
    elements = (
        SourceDeTension(1, 0, VS, sp.Symbol("I_S")),
        Admittance(1, 2, 1 / R1),
        Admittance(2, 0, S * C),
        Admittance(3, 0, 1 / R3),
        Admittance(3, 4, 1 / R2),
        *aop(2, 3, 4),
    )
    assert sp.simplify(sortie_ideale(elements, 4, 4) - attendue) == 0


def test_le_gain_a_un_pole_et_le_produit_gain_bande() -> None:
    """Planches 19 et 20 : `A_v(s) = a / (s + b)` ; CA3140, 100 dB et 4,5 MHz."""
    assert pytest.approx(1e5) == 10 ** (100 / 20)
    ouverte = bande_en_boucle_ouverte(4.5e6, 1e5)
    assert ouverte == pytest.approx(45.0)
    assert produit_gain_bande(1e5, ouverte) == pytest.approx(4.5e6)
    with pytest.raises(ValueError, match="gain en continu nul ou negatif"):
        bande_en_boucle_ouverte(4.5e6, 0.0)


def test_la_retroaction_conserve_le_produit_gain_bande() -> None:
    """Planches 22 a 26 : `A_cl(s) = 2 pi GBW / (s + 2 pi GBW (1/A_ol + beta))`."""
    produit, continu, retour = 4.5e6, 1e5, 0.01
    ferme = gain_ferme_a_un_pole(S, produit=produit, continu=continu, retour=retour)
    pole = -sp.solve(sp.denom(sp.together(ferme)), S)[0]
    bande = float(pole) / (2 * math.pi)
    assert bande == pytest.approx(produit * (1 / continu + retour))
    assert bande == pytest.approx(produit * retour, rel=2e-3)
    assert float(ferme.subs(S, 0)) == pytest.approx(1 / retour, rel=2e-3)
    assert float(ferme.subs(S, 0)) * bande == pytest.approx(produit)
    assert bande == pytest.approx(bande_apres_retroaction(continu, 45.0, retour))


def test_la_paire_differentielle_sature_en_courant() -> None:
    """Planche 34 : `I_x / I_1 = (e^x - e^-x) / (2 + e^x + e^-x)`, `x = V_id / v_T`.

    C'est `tanh(x / 2)`. A `3 v_T`, soit 75 mV, le courant atteint 90 %
    de `I_1` : la planche 72 le dit « ~ I_1 ».
    """
    assert pytest.approx(25e-3, rel=0.03) == TENSION_THERMIQUE
    for reduit in (-4.0, -0.5, 0.0, 1.0, 3.0):
        courant = courant_de_la_paire_differentielle(
            reduit * 25e-3, 2e-4, thermique=25e-3
        )
        imprimee = (math.exp(reduit) - math.exp(-reduit)) / (
            2 + math.exp(reduit) + math.exp(-reduit)
        )
        assert courant / 2e-4 == pytest.approx(imprimee)
        assert courant / 2e-4 == pytest.approx(math.tanh(reduit / 2))
    a_trois = courant_de_la_paire_differentielle(75e-3, 1.0, thermique=25e-3)
    assert a_trois == pytest.approx(0.905, abs=0.001)


def test_la_vitesse_de_balayage() -> None:
    """Planche 74 : `SR = I_1 / C_f`, borne de `dV_out / dt`."""
    balayage = vitesse_de_balayage(200e-6, 12e-12)
    assert balayage == pytest.approx(16.7e6, rel=2e-3)
    crete = 10.0
    limite = balayage / (2 * math.pi * crete)
    assert vitesse_de_balayage_minimale(limite, crete) == pytest.approx(balayage)
    with pytest.raises(ValueError, match="capacite de compensation nulle ou negative"):
        vitesse_de_balayage(200e-6, 0.0)


def test_les_courants_d_entree() -> None:
    """Planches 43 et 48 : `I_BIAS = (I_B+ + I_B-) / 2`, `I_OS = I_B+ - I_B-`."""
    assert courant_de_polarisation(0.9e-6, 0.7e-6) == pytest.approx(0.8e-6)
    assert courant_de_decalage(0.9e-6, 0.7e-6) == pytest.approx(0.2e-6)
    assert courant_de_decalage(0.7e-6, 0.7e-6) == 0.0


def test_le_mode_commun_et_les_alimentations() -> None:
    """Planches 59, 61 et 65 : `V_out = A_v (V_id + V_cm / CMRR + dV / PSRR)`."""
    assert tension_de_mode_commun(15.0 + 5e-6, 15.0) == pytest.approx(15.0, abs=1e-5)
    rejection = 10 ** (90 / 20)
    assert rejection_en_decibels(rejection) == pytest.approx(90.0)
    assert rejection_du_mode_commun(1e5, 1e5 / rejection) == pytest.approx(rejection)
    parfaite = sortie_avec_imperfections(1e5, 5e-6)
    assert parfaite == pytest.approx(0.5)
    commun = sortie_avec_imperfections(1e5, 5e-6, mode_commun=(1.0, rejection))
    assert commun - parfaite == pytest.approx(1e5 / rejection)
    rails = ((0.1, 1e4), (-0.2, 4e4))
    alimentee = sortie_avec_imperfections(1e5, 5e-6, alimentations=rails)
    assert alimentee - parfaite == pytest.approx(1e5 * (0.1 / 1e4 - 0.2 / 4e4))
    assert alimentee - parfaite == pytest.approx(0.5)
    with pytest.raises(ValueError, match="taux de rejection nul ou negatif"):
        sortie_avec_imperfections(1e5, 5e-6, mode_commun=(1.0, 0.0))


def test_les_cinq_montages_de_la_fin() -> None:
    """Planches 80 a 84 : courant-tension, soustracteur, additionneur, `-Z_2 / Z_1`."""
    assert convertisseur_courant_tension(10e3) == pytest.approx(-10e3)
    differentiel, commun = soustracteur(1e3, 10e3, 1e3, 10e3)
    assert differentiel == pytest.approx(10.0)
    assert commun == pytest.approx(0.0, abs=1e-12)
    assert somme_ponderee(10e3, (1e3, 1e3)) == pytest.approx((-10.0, -10.0))
    assert integrateur(1e3, 1e-6, S) == -1 / (1e3 * 1e-6 * S)
    assert derivateur(1e3, 1e-6, S) == -1e3 * 1e-6 * S
    assert sp.simplify(transfert_a_impedances(R, 1 / (S * C)) + 1 / (R * C * S)) == 0
    assert sp.simplify(transfert_a_impedances(1 / (S * C), R) + R * C * S) == 0
