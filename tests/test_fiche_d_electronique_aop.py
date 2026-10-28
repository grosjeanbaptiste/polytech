"""Fiche d'electronique : l'amplificateur operationnel et ses montages.

Articles 12 a 19 de `cursus/fiches/ElectroniqueInstrumentation.toml` : les
montages sont resolus par `tensions_a_aop_ideaux` (zero virtuel pose), puis
compares aux formules de la fiche et aux fonctions de `montages`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electronique.domain import montages
from polytech.electronique.domain.amplification import (
    diviseur_d_entree,
    gain_inverseur,
    gain_non_inverseur,
)
from polytech.electronique.domain.aop import saturer
from polytech.electronique.domain.aop_ideal import tensions_a_aop_ideaux
from polytech.electronique.domain.aop_reel import impedance_de_sortie_du_non_inverseur
from polytech.electronique.domain.boucle_fermee import (
    gain_en_boucle_fermee,
    impedance_d_entree_en_boucle_fermee,
    impedance_de_sortie_en_boucle_fermee,
)
from polytech.electronique.domain.defauts_aop import rejection_du_mode_commun
from polytech.electronique.domain.netliste import Admittance
from polytech.maths.domain.comparaison import egales, proches

S = sp.Symbol("s")
R1, R2, RF, R, C = sp.symbols("R_1 R_2 R_f R C", positive=True)


def _non_inverseur(z1: sp.Expr, z2: sp.Expr) -> sp.Expr:
    branches = [Admittance(2, 0, 1 / z1), Admittance(2, 3, 1 / z2)]
    return tensions_a_aop_ideaux(branches, [(1, 2, 3)], {1: 1}, 3)[2]


def _inverseur(z1: sp.Expr, z2: sp.Expr) -> sp.Expr:
    branches = [Admittance(1, 2, 1 / z1), Admittance(2, 3, 1 / z2)]
    return tensions_a_aop_ideaux(branches, [(0, 2, 3)], {1: 1}, 3)[2]


def test_le_non_inverseur_generalise_donne_un_plus_z2_sur_z1() -> None:
    """Article « Le non-inverseur » : `1 + Z2/Z1`, dont `1 + sRC` a condensateur."""
    assert egales(_non_inverseur(R1, R2), 1 + R2 / R1)
    assert egales(_non_inverseur(1 / (S * C), R), 1 + S * R * C)
    assert egales(_non_inverseur(R1 + 1 / (S * C), RF), 1 + RF / (R1 + 1 / (S * C)))
    assert sp.limit(_non_inverseur(R1 + 1 / (S * C), RF), S, 0) == 1
    assert egales(sp.limit(_non_inverseur(R1 + 1 / (S * C), RF), S, sp.oo), 1 + RF / R1)
    assert proches(gain_non_inverseur(1e3, 9e3), 10.0, 1e-12)


def test_l_inverseur_est_moins_z2_sur_z1_et_son_diviseur_d_entree() -> None:
    """Article « Le montage inverseur » : `-Z2/Z1` ; 1 k source devant 1 k : `-25`."""
    assert egales(_inverseur(R1, R2), -R2 / R1)
    gain = gain_inverseur(1e3, 50e3)
    assert gain == -50
    assert proches(gain * diviseur_d_entree(1e3, 1e3), -25, 1e-12)
    assert egales(_inverseur(R, 1 / (S * C)), montages.integrateur(R, C, S))


def test_le_suiveur_et_le_filtre_rc_entre_deux_suiveurs() -> None:
    """Articles « inverseur et suiveur » : `H = 1`, `H = 1/(sRC+1)` exactement."""
    charge = [Admittance(2, 0, 1 / R)]
    suiveur = tensions_a_aop_ideaux(charge, [(1, 2, 2)], {1: 1}, 2)
    assert suiveur[1] == 1
    entre_suiveurs = [Admittance(1, 2, 1 / R), Admittance(2, 0, S * C)]
    v = tensions_a_aop_ideaux(entre_suiveurs, [(2, 3, 3)], {1: 1}, 3)
    assert egales(v[2], 1 / (S * R * C + 1))


def test_l_impedance_de_sortie_du_non_inverseur_est_tres_faible() -> None:
    """Article « Le montage non-inverseur » : `Z_out ~ (R_o/A_v)(1 + R_2/R_1)`."""
    sortie = impedance_de_sortie_du_non_inverseur(
        100.0, 1e6, vers_la_masse=1e3, contre_reaction=9e3, entree=1e12
    )
    assert proches(sortie, 100 / 1e6 * (1 + 9), 1e-6)


def test_les_blocs_a_amplificateur_operationnel() -> None:
    """Article « Reconnaitre les blocs » : additionneur, differentiel, derivateur."""
    assert montages.somme_ponderee(10e3, (1e3, 2e3, 5e3)) == (-10, -5, -2)
    assert montages.convertisseur_courant_tension(2e3) == -2e3
    assert egales(montages.derivateur(R, C, S), -S * R * C)
    assert egales(montages.integrateur(R, C, S), -1 / (S * R * C))
    differentiel = montages.soustracteur(1e3, 10e3, 1e3, 10e3)
    assert proches(differentiel[0], 10, 1e-9)
    assert proches(differentiel[1], 0, 1e-9)


def test_le_gain_boucle_ferme_tend_vers_un_sur_beta_et_l_impedance_se_corrige() -> None:
    """Article « La retroaction » : `A/(1 + beta A)`, `Z_in (1+bA)`, `Z_out/(1+bA)`."""
    a, beta = 1e5, 0.1
    assert proches(gain_en_boucle_fermee(a, beta), 1 / beta, 1e-3)
    assert impedance_d_entree_en_boucle_fermee(1e6, a, beta) == 1e6 * (1 + a * beta)
    assert proches(
        impedance_de_sortie_en_boucle_fermee(100, a, beta), 100 / (1 + a * beta), 1e-12
    )


def test_le_cmrr_ramene_le_mode_commun_a_l_entree() -> None:
    """Article « Le mode commun » : `A_v (V_id + V_cm / CMRR)`, `CMRR = A_v/A_cm`."""
    a_v, a_cm, v_id, v_cm = 1e5, 10.0, 1e-3, 2.0
    cmrr = rejection_du_mode_commun(a_v, a_cm)
    assert cmrr == 1e4
    assert proches(a_v * v_id + a_cm * v_cm, a_v * (v_id + v_cm / cmrr), 1e-9)


def test_l_amplificateur_sature_sur_ses_rails() -> None:
    """Article « Le modele » : `V_out = A_v (V+ - V-)` borne a `+-V_alim`."""
    assert saturer(1e5 * 1e-3, 12) == 12
    assert saturer(1e5 * -2e-3, 12) == -12
    assert saturer(1e5 * 5e-5, 12) == 5
    assert proches(12 / 200, 0.06, 1e-12)
