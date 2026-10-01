"""Le chapitre 1 partie III : l'analyse des circuits par Laplace.

Ce que la planche enseigne, et que le module ne portait pas : les
impedances complexes et leurs associations, les deux diviseurs, la
decomposition en fractions simples dans ses **trois** cas -- poles simples,
poles reels multiples, poles conjugues -- et les reponses impulsionnelle
et indicielle qui s'en deduisent.

Le fil directeur du chapitre est une equivalence : passer des equations
integro-differentielles aux impedances complexes. C'est elle qui rend
l'analyse algebrique, et c'est elle que les tests verifient -- une
impedance n'est juste que si elle redonne l'equation dont elle vient.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.domain.laplace import (
    decomposition_en_fractions_simples,
    diviseur_de_courant,
    diviseur_de_tension,
    est_stable,
    impedance_de_bobine,
    impedance_de_condensateur,
    impedance_de_resistance,
    impedances_en_parallele,
    impedances_en_serie,
    multiplicites_des_poles,
    reponse_a_un_echelon,
    reponse_impulsionnelle,
    residu_en_un_pole,
)

P = sp.Symbol("p")
T = sp.Symbol("t", positive=True)
R, L, C = sp.symbols("R L C", positive=True)


def test_les_trois_impedances_complexes_du_chapitre() -> None:
    """`R`, `p L`, `1 / (p C)` : la derivation devient une multiplication."""
    assert impedance_de_resistance(R, P) == R
    assert sp.simplify(impedance_de_bobine(L, P) - P * L) == 0
    assert sp.simplify(impedance_de_condensateur(C, P) - 1 / (P * C)) == 0


def test_l_impedance_de_la_bobine_redonne_son_equation_integro_differentielle() -> None:
    """`u = L di/dt` devient `U = p L I`, **a conditions initiales nulles**.

    La verification n'est pas formelle. On prend un courant, on calcule la
    tension par l'equation temporelle, on transforme les deux, et le
    rapport doit etre l'impedance.

    La condition initiale n'est pas une precaution d'ecriture : la
    transformee exacte est `U = p L I - L i(0)`, et avec `i(0) = 1` le
    rapport vaut `-L` au lieu de `p L`. Le courant choisi s'annule donc en
    zero, comme le chapitre le suppose partout -- et c'est precisement ce
    que l'analyse par impedances abandonne par rapport a Laplace.
    """
    courant = 1 - sp.exp(-T)
    tension = L * sp.diff(courant, T)
    image_courant = sp.laplace_transform(courant, T, P, noconds=True)
    image_tension = sp.laplace_transform(tension, T, P, noconds=True)
    rapport = sp.simplify(image_tension / image_courant)
    assert sp.simplify(rapport - impedance_de_bobine(L, P)) == 0


def test_les_impedances_en_serie_s_additionnent() -> None:
    serie = impedances_en_serie(R, P * L)
    assert sp.simplify(serie - (R + P * L)) == 0


def test_les_admittances_s_additionnent_en_parallele() -> None:
    parallele = impedances_en_parallele(R, R)
    assert sp.simplify(parallele - R / 2) == 0


def test_une_association_vide_est_refusee() -> None:
    """Zero impedance en serie rendrait zero, soit un court-circuit."""
    with pytest.raises(ValueError, match="aucune impedance"):
        impedances_en_serie()


def test_le_diviseur_de_tension_generalise() -> None:
    """`Z_2 / (Z_1 + Z_2)` : la forme que le chapitre generalise aux impedances."""
    rapport = diviseur_de_tension(R, impedance_de_condensateur(C, P))
    attendu = 1 / (1 + P * R * C)
    assert sp.simplify(rapport - attendu) == 0


def test_le_diviseur_de_courant_est_le_dual_du_diviseur_de_tension() -> None:
    """Il partage le courant a l'inverse des impedances."""
    rapport = diviseur_de_courant(R, R)
    assert sp.simplify(rapport - sp.Rational(1, 2)) == 0


def test_un_diviseur_de_tension_sans_charge_est_refuse() -> None:
    with pytest.raises(ValueError, match="impedance nulle"):
        diviseur_de_tension(sp.Integer(0), sp.Integer(0))


def test_les_multiplicites_des_poles_sont_rendues() -> None:
    """Le chapitre distingue trois cas, et la multiplicite les separe."""
    simple = multiplicites_des_poles(1 / ((P + 1) * (P + 2)), P)
    assert simple == {sp.Integer(-1): 1, sp.Integer(-2): 1}
    double = multiplicites_des_poles(1 / (P + 1) ** 2, P)
    assert double == {sp.Integer(-1): 2}


def test_la_decomposition_a_poles_simples_redonne_la_fraction() -> None:
    """Premier cas du chapitre : poles de multiplicite un."""
    fraction = 1 / ((P + 1) * (P + 2))
    termes = decomposition_en_fractions_simples(fraction, P)
    assert sp.simplify(sum(termes) - fraction) == 0
    attendu = 2
    assert len(termes) == attendu


def test_la_decomposition_a_poles_reels_multiples_redonne_la_fraction() -> None:
    """Deuxieme cas : un pole de multiplicite deux donne deux termes."""
    fraction = 1 / ((P + 1) ** 2 * (P + 3))
    termes = decomposition_en_fractions_simples(fraction, P)
    assert sp.simplify(sum(termes) - fraction) == 0


def test_la_decomposition_a_poles_conjugues_redonne_la_fraction() -> None:
    """Troisieme cas : des poles complexes conjugues."""
    fraction = 1 / (P**2 + 2 * P + 5)
    termes = decomposition_en_fractions_simples(fraction, P)
    assert sp.simplify(sum(termes) - fraction) == 0


def test_le_residu_en_un_pole_simple_vaut_la_limite_du_produit() -> None:
    """`lim (p - p_i) F(p)` : la recette que le chapitre donne."""
    fraction = 1 / ((P + 1) * (P + 2))
    assert sp.simplify(residu_en_un_pole(fraction, P, sp.Integer(-1)) - 1) == 0


def test_un_residu_hors_des_poles_est_nul() -> None:
    """Et le dire evite de croire a un terme qui n'existe pas."""
    fraction = 1 / ((P + 1) * (P + 2))
    assert sp.simplify(residu_en_un_pole(fraction, P, sp.Integer(0))) == 0


def test_la_reponse_impulsionnelle_a_poles_simples() -> None:
    """`1/(p+1)` a pour original `e^{-t}`."""
    reponse = reponse_impulsionnelle(1 / (P + 1), P, T)
    assert sp.simplify(reponse - sp.exp(-T)) == 0


def test_la_reponse_impulsionnelle_a_pole_double_porte_un_facteur_t() -> None:
    """`1/(p+1)^2` donne `t e^{-t}` : c'est ce que la multiplicite ajoute."""
    reponse = reponse_impulsionnelle(1 / (P + 1) ** 2, P, T)
    assert sp.simplify(reponse - T * sp.exp(-T)) == 0


def test_la_reponse_impulsionnelle_a_poles_conjugues_oscille() -> None:
    """`1/(p^2+1)` donne `sin t`."""
    reponse = reponse_impulsionnelle(1 / (P**2 + 1), P, T)
    assert sp.simplify(reponse - sp.sin(T)) == 0


def test_la_reponse_indicielle_est_l_integrale_de_l_impulsionnelle() -> None:
    """Recoupement : deux chemins pour la meme reponse.

    L'une passe par `T(p)/p`, l'autre integre la reponse impulsionnelle.
    Les deux doivent concorder, et aucune n'emprunte a l'autre.
    """
    transfert = 1 / (P + 1)
    par_laplace = reponse_a_un_echelon(transfert, sp.Integer(1), P, T)
    impulsionnelle = reponse_impulsionnelle(transfert, P, T)
    variable = sp.Symbol("theta", positive=True)
    par_integration = sp.integrate(impulsionnelle.subs(T, variable), (variable, 0, T))
    assert sp.simplify(par_laplace - par_integration) == 0


def test_la_stabilite_se_lit_sur_les_poles() -> None:
    assert est_stable(1 / (P + 1), P)
    assert not est_stable(1 / (P - 1), P)
