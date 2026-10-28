"""Les criteres de comparaison que les tests des courriers empruntent a la lib."""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from polytech.maths.domain.comparaison import (
    argument_en_degres,
    dephasage_en_degres,
    egales,
    egales_arrondies,
    eloignees,
    en_degres,
    est_negative,
    est_positive,
    exponentielle,
    proches,
    proches_relativement,
    rapport_au_dessus,
    suite_nulle,
    suites_proches,
)

X = sp.Symbol("x", real=True)
A = sp.Symbol("a", positive=True)


def test_deux_expressions_egales_sous_une_identite_trigonometrique() -> None:
    assert egales(sp.sin(X) ** 2 + sp.cos(X) ** 2, 1)


def test_deux_expressions_distinctes_ne_sont_pas_egales() -> None:
    assert not egales(sp.sin(X), sp.cos(X))


def test_une_valeur_numerique_exacte_est_egale_a_son_ecriture_exacte() -> None:
    assert egales(sp.sin(sp.pi / 2 + sp.atan(sp.Rational(1, 2))), 2 / sp.sqrt(5))


def test_un_ecart_infime_mais_non_nul_est_refuse() -> None:
    assert not egales(X + sp.Rational(1, 10**6), X)


def test_les_valeurs_absolues_non_simplifiees_sont_egales_en_echantillons() -> None:
    assert egales(sp.Abs(X) ** 2, X**2)
    assert egales(sp.sqrt(X**2), sp.Abs(X))
    assert not egales(sp.sqrt(X**2), X)


def test_un_symbole_positif_n_est_echantillonne_qu_en_positif() -> None:
    assert egales(sp.sqrt(A**2), A)


def test_deux_matrices_sont_egales_terme_a_terme() -> None:
    identite = sp.Matrix([[sp.sin(X) ** 2 + sp.cos(X) ** 2, X - X], [0, 1]])
    assert egales(identite, sp.Matrix([[1, 0], [0, 1]]))
    assert not egales(identite, sp.Matrix([[1, 0], [1, 1]]))


def test_deux_matrices_de_formes_differentes_ne_sont_pas_egales() -> None:
    assert not egales(sp.zeros(2, 2), sp.zeros(2, 3))


def test_un_nombre_complexe_est_compare_par_son_ecart() -> None:
    assert egales(sp.exp(sp.I * sp.pi), -1)
    assert not egales(sp.I, -sp.I)


def test_une_expression_non_evaluable_leve_au_lieu_de_conclure() -> None:
    f = sp.Function("f")
    with pytest.raises(TypeError):
        egales(f(X), 0)


def test_proches_est_inclusif_a_la_tolerance() -> None:
    assert proches(1.0, 1.5, 0.5)
    assert not proches(1.0, 1.5, 0.49)
    assert proches(1.5, 1.0, 0.5)


def test_eloignees_est_strict() -> None:
    assert eloignees(1.0, 1.5, 0.49)
    assert not eloignees(1.0, 1.5, 0.5)


@pytest.mark.parametrize("tolerance", [-1.0, math.nan])
def test_une_tolerance_negative_ou_indefinie_leve(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        proches(1.0, 1.0, tolerance)
    with pytest.raises(ValueError, match="tolerance"):
        eloignees(1.0, 1.0, tolerance)


def test_les_radians_se_lisent_en_degres() -> None:
    assert en_degres(math.pi) == pytest.approx(180.0, abs=1e-12)
    assert en_degres(-math.pi / 2) == pytest.approx(-90.0, abs=1e-12)
    assert en_degres(0.0) == 0.0


def test_l_argument_en_degres_d_un_complexe() -> None:
    assert argument_en_degres(1j) == pytest.approx(90.0, abs=1e-12)
    assert argument_en_degres(-1 + 0j) == pytest.approx(180.0, abs=1e-12)
    assert argument_en_degres(1 - 1j) == pytest.approx(-45.0, abs=1e-12)


def test_l_argument_en_degres_d_un_exact_sympy() -> None:
    assert argument_en_degres(1 + sp.sqrt(3) * sp.I) == pytest.approx(60.0, abs=1e-12)


def test_l_argument_de_zero_leve() -> None:
    with pytest.raises(ValueError, match="zero"):
        argument_en_degres(0j)


def test_le_dephasage_est_l_argument_du_rapport() -> None:
    assert dephasage_en_degres(1j, 1 + 0j) == pytest.approx(90.0, abs=1e-12)
    assert dephasage_en_degres(1 + 0j, 1j) == pytest.approx(-90.0, abs=1e-12)
    assert dephasage_en_degres(2 + 2j, 1 + 0j) == pytest.approx(45.0, abs=1e-12)


def test_une_identite_sur_des_fonctions_non_definies_se_prouve_symboliquement() -> None:
    y = sp.Symbol("y", real=True)
    f, g = sp.Function("f"), sp.Function("g")
    u = f(X + y) + g(X - y)
    ecart = sp.diff(u, X, 2).doit() - sp.diff(u, y, 2).doit()
    assert egales(ecart, 0)


def test_proches_relativement_mesure_l_ecart_a_la_reference() -> None:
    assert proches_relativement(1e6 + 1, 1e6, 2e-6)
    assert not proches_relativement(1e6 + 1, 1e6, 5e-7)
    assert proches_relativement(-100.0, -101.0, 0.01)
    assert not proches_relativement(3.0, 1.0, 1.5)  # relatif a la reference


def test_proches_relativement_refuse_une_reference_nulle_non_atteinte() -> None:
    assert proches_relativement(0.0, 0.0, 1e-3)
    assert not proches_relativement(1e-9, 0.0, 1e-3)


def test_proches_relativement_leve_pour_une_tolerance_invalide() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        proches_relativement(1.0, 1.0, -1e-3)


def test_egales_arrondies_compare_au_nombre_de_decimales_dit() -> None:
    assert egales_arrondies(394.46, 394.5, 1)
    assert not egales_arrondies(394.46, 394.4, 1)
    assert egales_arrondies(949.7, 950, 0)
    assert egales_arrondies(-0.0004, 0.0, 3)


def test_egales_arrondies_refuse_des_decimales_negatives() -> None:
    with pytest.raises(ValueError, match="decimales"):
        egales_arrondies(1.0, 1.0, -1)


def test_rapport_au_dessus_est_strict() -> None:
    assert rapport_au_dessus(30.0, 3.0, 9.9)
    assert not rapport_au_dessus(30.0, 3.0, 10.0)
    assert rapport_au_dessus(-30.0, -3.0, 9.9)


def test_rapport_au_dessus_leve_pour_une_reference_nulle() -> None:
    with pytest.raises(ZeroDivisionError):
        rapport_au_dessus(1.0, 0.0, 1.0)


def test_exponentielle_donne_e_puissance_x() -> None:
    assert exponentielle(0.0) == 1.0
    assert exponentielle(1.0) == pytest.approx(2.718281828459045, abs=1e-15)
    assert exponentielle(-90.0) == pytest.approx(8.194e-40, rel=1e-3)


def test_deux_suites_sont_proches_terme_a_terme() -> None:
    assert suites_proches((1.0, 2.0), [1.04, 1.96], 0.05)
    assert not suites_proches((1.0, 2.0), (1.0, 2.06), 0.05)


def test_deux_suites_sont_proches_en_relatif_si_on_le_demande() -> None:
    assert suites_proches((100.0, 1.0), (101.0, 1.01), 0.011, relativement=True)
    assert not suites_proches((100.0, 1.0), (101.0, 1.5), 0.011, relativement=True)


def test_deux_suites_de_longueurs_differentes_levent() -> None:
    with pytest.raises(ValueError, match="longueurs"):
        suites_proches((1.0,), (1.0, 2.0), 0.1)
    with pytest.raises(ValueError, match="longueurs"):
        suites_proches((1.0, 2.0), (1.0,), 0.1)


def test_deux_suites_vides_sont_proches() -> None:
    assert suites_proches((), (), 0.1)


def test_le_signe_d_une_expression_se_lit_apres_simplification() -> None:
    assert est_positive(sp.sqrt(A**2) + A)
    assert not est_positive(-A)
    assert est_negative(-A * A)
    assert not est_negative(A)


def test_un_signe_indetermine_n_est_ni_positif_ni_negatif() -> None:
    assert not est_positive(X)
    assert not est_negative(X)
    assert not est_positive(0)
    assert not est_negative(0)


def test_des_suites_numpy_se_comparent_comme_des_listes() -> None:
    assert suites_proches(np.array([0.0, 1e-13]), np.zeros(2), 1e-12)
    assert not suites_proches(np.array([0.0, 1e-3]), np.zeros(2), 1e-12)


def test_une_suite_nulle_a_la_tolerance_pres() -> None:
    assert suite_nulle(np.array([0.0, -1e-13, 5e-13]), 1e-12)
    assert not suite_nulle([0.0, 2e-12], 1e-12)
    assert suite_nulle([], 1e-12)


def test_deux_rationnels_distincts_de_un_epsilon_sont_distingues_exactement() -> None:
    """Un ecart exact de `10^-13` passerait sous 1e-12 en numerique."""
    assert not egales(1, 1 + sp.Rational(1, 10**13))
    x = sp.Symbol("x")
    assert not egales(x + 1, x + 1 + sp.Rational(1, 10**13))


def test_un_ecart_symboliquement_nul_est_egal_sans_echantillonner() -> None:
    x = sp.Symbol("x")
    assert egales(sp.sin(x) ** 2 + sp.cos(x) ** 2, 1)


def test_un_ecart_flottant_negligeable_reste_egal() -> None:
    """Dernier recours numerique : 0,1 + 0,2 et 0,3 different de 5e-17."""
    assert egales(0.1 + 0.2, 0.3)
