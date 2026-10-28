"""Figures d'electronique dont une formule porte un coefficient : relues par la lib."""

from __future__ import annotations

import re

import pytest
import sympy as sp

from polytech.electronique.domain.bruit_thermique import densite_thermique
from polytech.electronique.domain.echelle_lc import elements_de_butterworth
from polytech.electronique.domain.facteur_de_bruit import (
    TEMPERATURE_DE_REFERENCE as T0,
)
from polytech.electronique.domain.facteur_de_bruit import figure_de_bruit
from polytech.electronique.domain.modele_tension_courant import (
    densite_ramenee_a_l_entree,
)
from polytech.electronique.domain.pole_zero import pente_asymptotique
from polytech.maths.domain.comparaison import egales, proches
from polytech.thermique.domain.rayonnement.corps_noir import BOLTZMANN
from tests.fiches_toml import cherche, figure

M = "ElectroniqueInstrumentation"
COUVERTES = {
    "gabarit_et_carte_des_poles_et_zeros",
    "bruit_thermique",
    "modele_de_bruit_en_tension_et_en_courant",
    "facteur_et_figure_de_bruit",
    "prototype_en_echelle_lc",
}
S = sp.Symbol("s")


def test_la_carte_des_poles_et_des_zeros_donne_la_pente_de_la_figure() -> None:
    texte = figure(M, "gabarit_et_carte_des_poles_et_zeros")
    croix = re.findall(r"red\] at \((-?[\d.]+),(-?[\d.]+)\) \{\$\\times\$\}", texte)
    poles = [sp.nsimplify(complex(float(x), float(y))) for x, y in croix]
    cercles = re.findall(r"\(0,(-?[\d.]+)\) circle", texte)
    zeros = [sp.I * sp.nsimplify(float(y)) for y in cercles]
    coefficient = int(cherche(r"(\d+)\\,\(n_z - n_p\)", texte, 1))
    assert (len(poles), len(zeros)) == (3, 2)
    assert coefficient == -pente_asymptotique(0, 1)
    pente = pente_asymptotique(len(zeros), len(poles))
    assert pente == coefficient * (len(zeros) - len(poles))
    h = sp.prod([S - z for z in zeros]) / sp.prod([S - p for p in poles])
    assert egales(h.subs(S, zeros[0]), 0)


def test_le_bruit_thermique_de_la_figure_est_quatre_k_t_r() -> None:
    coefficient = int(cherche(r"S_e = (\d) k_B T R", figure(M, "bruit_thermique"), 1))
    assert densite_thermique(1e3, 300.0) == pytest.approx(
        coefficient * BOLTZMANN * 300.0 * 1e3
    )


def test_le_bruit_ramene_a_l_entree_de_la_figure() -> None:
    texte = figure(M, "modele_de_bruit_en_tension_et_en_courant")
    coefficient = int(cherche(r"\+ (\d)k_BTR_s", texte, 1))
    assert "e_n^2 + (i_n R_s)^2" in texte
    sv, si, rs = 4e-18, 1e-24, 2e3
    attendu = sv + rs**2 * si + coefficient * BOLTZMANN * T0 * rs
    assert proches(densite_ramenee_a_l_entree(rs, sv, si), attendu, 1e-30)


def test_la_figure_de_bruit_est_dix_log_f() -> None:
    coefficient = int(
        cherche(r"NF = (\d+)\\log F", figure(M, "facteur_et_figure_de_bruit"), 1)
    )
    assert figure_de_bruit(10.0) == pytest.approx(coefficient * 1.0)
    assert figure_de_bruit(2.0) == pytest.approx(coefficient * 0.30103, abs=1e-4)


def test_le_prototype_de_la_figure_a_autant_d_elements_que_son_ordre() -> None:
    texte = figure(M, "prototype_en_echelle_lc")
    ordre = int(cherche(r"d'ordre (\d)", texte, 1))
    elements = re.findall(r"l=\$([LC])_(\d)\$", texte)
    assert ordre == len(elements) == len(elements_de_butterworth(ordre))
    par_rang = sorted(elements, key=lambda element: int(element[1]))
    assert [type_ for type_, _ in par_rang] == ["L", "C", "L"]
