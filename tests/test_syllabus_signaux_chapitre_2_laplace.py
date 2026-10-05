"""Le syllabus de signaux et systemes, chapitre 2 : l'analyse operationnelle.

Source : `SignauxSystemes/SYLLABUS/Syllabus_2019_Chapitre_02_Transformees.pdf`,
section 2.2 (pages 24 a 33), lue sur les pages rendues.

Les oracles sont le tableau 2.2, les proprietes (2.24) a (2.32) et les
exemples 2.13 a 2.16, avec les sorties de `residue` que le syllabus
imprime : `r = [-1, 1]`, `r = [0.5 - 0.25j, 0.5 + 0.25j]`, `r = [3, -6]`.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from polytech.maths.domain.laplace import proprietes, theoremes
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE as P_DES_MATHS
from polytech.maths.domain.symboles import TEMPS as T_DES_MATHS
from polytech.signaux.domain.composantes import nature_de_la_composante
from polytech.signaux.domain.heaviside import (
    division,
    original,
    residu,
    residus_d_un_pole_double,
)
from polytech.signaux.domain.modes import residus
from polytech.signaux.domain.signaux_usuels import (
    ALPHA,
    OMEGA,
    SIGMA,
    SIGNAUX_USUELS,
    A,
    B,
    Paire,
)
from polytech.signaux.domain.symboles import IMAGE, TEMPS
from polytech.signaux.domain.transformees import transformee_de_laplace

P, T = IMAGE, TEMPS
VALEURS = {A: 2, B: 5, SIGMA: 1, OMEGA: 3, ALPHA: 4}
"""Des valeurs d'essai, avec `sigma < rho` : poles complexes conjugues."""


def test_le_tableau_2_2_porte_treize_signaux() -> None:
    assert len(SIGNAUX_USUELS) == 13
    assert all(isinstance(paire, Paire) for paire in SIGNAUX_USUELS.values())
    assert SIGNAUX_USUELS["impulsion de Dirac"].image == 1
    assert SIGNAUX_USUELS["echelon"].image == 1 / P
    assert SIGNAUX_USUELS["rampe"].image == 1 / P**2


@pytest.mark.parametrize("nom", sorted(SIGNAUX_USUELS))
def test_chaque_ligne_du_tableau_2_2_est_une_paire_de_laplace(nom: str) -> None:
    """La transformee de l'original, calculee, est l'image du tableau."""
    paire = SIGNAUX_USUELS[nom]
    calculee = transformee_de_laplace(paire.original.subs(VALEURS))
    ecart = sp.lambdify(P, calculee - paire.image.subs(VALEURS), "numpy")
    points = np.array([0.7, 2.0 + 1.5j, 6.0 - 2.0j])
    assert np.allclose(ecart(points), 0, atol=1e-10)


def test_les_poles_reels_du_tableau_quand_sigma_depasse_rho() -> None:
    """Deux dernieres lignes : `p^2 + 2 sigma p + rho^2 = (p + a)(p + b)`."""
    paire = SIGNAUX_USUELS["difference de deux exponentielles"]
    denominateur = sp.expand(1 / paire.image)
    assert denominateur.coeff(P, 1) == A + B
    assert denominateur.coeff(P, 0) == A * B


def test_la_derivee_fait_entrer_la_condition_initiale() -> None:
    """(2.25) : `L[df/dt] = p F(p) - f(0-)`, sur `cos(3 t)`."""
    signal = sp.cos(3 * T_DES_MATHS)
    image = Transformee.de(signal).image
    derivee = Transformee.de(sp.diff(signal, T_DES_MATHS)).image
    attendue = proprietes.de_la_derivee_n(image, [signal.subs(T_DES_MATHS, 0)])
    assert sp.simplify(derivee - attendue) == 0


def test_l_integrale_divise_par_p() -> None:
    """(2.26), sur `exp(-2 t)` : son integrale est `(1 - exp(-2 t)) / 2`."""
    image = Transformee.de(sp.exp(-2 * T_DES_MATHS)).image
    integrale = Transformee.de((1 - sp.exp(-2 * T_DES_MATHS)) / 2).image
    assert sp.simplify(integrale - proprietes.de_l_integrale(image)) == 0
    assert sp.simplify(proprietes.de_l_integrale(image) - image / P_DES_MATHS) == 0


def test_le_retard_et_la_translation() -> None:
    """(2.27) et (2.28)."""
    image = 1 / P_DES_MATHS**2
    retardee = proprietes.du_retard(image, 5)
    assert sp.simplify(retardee - sp.exp(-5 * P_DES_MATHS) * image) == 0
    amortie = proprietes.du_produit_par_une_exponentielle(image, -3)
    assert sp.simplify(amortie - 1 / (P_DES_MATHS + 3) ** 2) == 0


def test_la_convolution_devient_un_produit() -> None:
    """(2.29)."""
    premiere, seconde = 1 / P_DES_MATHS, 1 / (P_DES_MATHS + 1)
    assert proprietes.de_la_convolution(premiere, seconde) == premiere * seconde


def test_les_theoremes_de_la_valeur_initiale_et_de_la_valeur_finale() -> None:
    """(2.30) et (2.31), sur `2 - exp(-t)` : 1 au depart, 2 a l'arrivee."""
    image = 2 / P_DES_MATHS - 1 / (P_DES_MATHS + 1)
    assert theoremes.valeur_initiale(image) == 1
    assert theoremes.valeur_finale(image) == 2


def test_la_periodification() -> None:
    """(2.32) : `F(p) / (1 - exp(-p T))`."""
    motif = 1 / P_DES_MATHS
    periodique = proprietes.d_une_fonction_periodique(motif, 2)
    assert sp.simplify(periodique - motif / (1 - sp.exp(-2 * P_DES_MATHS))) == 0


def test_l_exemple_2_13_deux_poles_simples() -> None:
    """`1 / (p^2 + 3 p + 2)` : residus 1 et -1, `exp(-t) - exp(-2 t)`."""
    image = 1 / (P**2 + 3 * P + 2)
    assert residu(image, -1) == 1
    assert residu(image, -2) == -1
    assert sp.simplify(original(image) - (sp.exp(-T) - sp.exp(-2 * T))) == 0
    valeurs, poles = residus([1.0], [1.0, 3.0, 2.0])
    assert sorted(zip(poles.real, valeurs.real, strict=True)) == pytest.approx(
        [(-2, -1), (-1, 1)]
    )


def test_l_exemple_2_14_deux_poles_imaginaires_conjugues() -> None:
    """`(p + 1) / (p^2 + 4)` : residu `1/2 - j/4`, `cos(2 t) + sin(2 t) / 2`."""
    image = (P + 1) / (P**2 + 4)
    amplitude = residu(image, 2 * sp.I)
    assert amplitude == sp.Rational(1, 2) - sp.I / 4
    assert residu(image, -2 * sp.I) == sp.conjugate(amplitude)
    attendu = sp.cos(2 * T) + sp.sin(2 * T) / 2
    assert sp.simplify(original(image) - attendu) == 0
    phaseur = complex(2 * amplitude)
    assert abs(phaseur) == pytest.approx(1.1180, abs=1e-4)
    assert np.angle(phaseur) == pytest.approx(-0.4636, abs=1e-4)


def test_l_exemple_2_15_un_pole_double() -> None:
    """`3 p / (p + 2)^2` : `A_2 = -6`, `A_1 = 3`."""
    image = 3 * P / (P**2 + 4 * P + 4)
    assert residus_d_un_pole_double(image, -2) == (3, -6)
    attendu = -6 * T * sp.exp(-2 * T) + 3 * sp.exp(-2 * T)
    assert sp.simplify(original(image) - attendu) == 0


def test_l_exemple_2_16_une_fraction_quelconque() -> None:
    """Le quotient `p + 1` donne `delta'(t) + delta(t)`."""
    image = (P**3 + 3 * P**2 + 4 * P + 3) / (P**2 + 2 * P + 1)
    quotient, reste = division(image)
    assert quotient == P + 1
    assert sp.simplify(reste - (P + 2) / (P + 1) ** 2) == 0
    impulsions = sp.DiracDelta(T, 1) + sp.DiracDelta(T)
    attendu = impulsions + T * sp.exp(-T) + sp.exp(-T)
    assert sp.simplify(original(image) - attendu) == 0


def test_le_quotient_range_les_impulsions_par_ordre_de_derivation() -> None:
    """(2.41) : `q_0 + q_1 p` donne `q_0 delta(t) + q_1 delta'(t)`."""
    image = 2 * P + 3 + 1 / (P + 1)
    attendu = 3 * sp.DiracDelta(T) + 2 * sp.DiracDelta(T, 1) + sp.exp(-T)
    assert sp.simplify(original(image) - attendu) == 0


def test_un_pole_triple_sort_du_cours() -> None:
    """Note 11 : « nous ne considererons pas les poles de multiplicite superieure »."""
    with pytest.raises(ValueError, match="multiplicite superieure a deux"):
        original(1 / (P + 1) ** 3)
    with pytest.raises(ValueError, match="n'est pas un pole double"):
        residus_d_un_pole_double(1 / (P + 1), -1)
    with pytest.raises(ValueError, match="n'est pas un pole simple"):
        residu(1 / (P + 1), -3)


def test_la_forme_de_chaque_composante_se_lit_sur_son_pole() -> None:
    """Section 2.2.5 et fig. 2.25."""
    assert nature_de_la_composante(0) == "terme constant"
    assert nature_de_la_composante(2j) == "sinusoide non amortie"
    assert nature_de_la_composante(-1 + 2j) == "sinusoide amortie"
    assert nature_de_la_composante(1 - 2j) == "sinusoide amplifiee"
    assert nature_de_la_composante(-3) == "exponentielle decroissante"
    assert nature_de_la_composante(3) == "exponentielle croissante"
    double = nature_de_la_composante(-3, double=True)
    assert double == "exponentielle decroissante, multipliee par t"
    assert nature_de_la_composante(0, double=True) == "terme constant, multiplie par t"
