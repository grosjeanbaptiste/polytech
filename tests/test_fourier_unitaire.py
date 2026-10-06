"""La transformee de Fourier dans la convention unitaire de Mathematiques III.

Oracles : `MathematiquesIngenieur3/SLIDES/
Slides_11_Transformees_partie_1_Transformee_de_Fourier.pdf`, 19 planches.
Chaque enonce est verifie en integrant la definition telle que la planche
l'ecrit, sans passer par la convention du cours du soir.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.fourier.proprietes import energie_temporelle
from polytech.maths.domain.fourier.transformee import (
    PULSATION,
    integrale_sur_la_droite,
)
from polytech.maths.domain.fourier.unitaire import (
    de_la_convolution,
    en_cosinus,
    en_sinus,
    energie_spectrale,
    inverse_unitaire,
    transformee_unitaire,
)
from polytech.maths.domain.symboles import VARIABLE

X, S = VARIABLE, PULSATION
RACINE = sp.sqrt(2 * sp.pi)
PAIRE = sp.exp(-sp.Abs(X))
CAUSALE = sp.exp(-X) * sp.Heaviside(X)
"""Ni paire ni impaire : un signe faux au noyau la trahit."""
PORTE = sp.Piecewise((1, sp.Abs(X) <= sp.Rational(1, 2)), (0, True))
TRIANGLE = sp.Piecewise((1 - sp.Abs(X), sp.Abs(X) <= 1), (0, True))


def _nul(ecart: sp.Expr) -> bool:
    return bool(sp.simplify(ecart.rewrite(sp.cos)) == 0)


def test_la_definition_de_la_planche_8() -> None:
    """Planche 8, (1) : `(1 / racine(2 pi)) int f(x) e^(-i s x) dx`."""
    directe = integrale_sur_la_droite(CAUSALE * sp.exp(-sp.I * S * X), X) / RACINE
    assert _nul(transformee_unitaire(CAUSALE) - directe)
    assert _nul(directe - 1 / (RACINE * (1 + sp.I * S)))
    assert _nul(transformee_unitaire(PAIRE) - sp.sqrt(2 / sp.pi) / (1 + S**2))


def test_la_transformee_inverse_de_la_planche_11_rend_la_fonction() -> None:
    """Planche 11 : `F^-1(F(f)) = f` en tout point de continuite."""
    assert _nul(inverse_unitaire(transformee_unitaire(PAIRE)) - PAIRE)


def test_la_fonction_de_la_planche_19_est_l_image_de_son_image() -> None:
    """Planche 19, « le cas de la fonction 1 / (1 + x2) » : par la dualite.

    `F(e^(-|x|)) = racine(2 / pi) / (1 + s2)` ; la transformee inverse, qui
    ne differe de la directe que par le signe du noyau, rend donc
    `racine(pi / 2) e^(-|x|)` pour `1 / (1 + s2)`.
    """
    image = inverse_unitaire(1 / (1 + S**2))
    assert _nul(image - sp.sqrt(sp.pi / 2) * PAIRE)


def test_le_cas_d_une_fonction_paire_de_la_planche_12() -> None:
    """Planche 12 : `F(f) = racine(2 / pi) int de 0 a l'infini de f cos(s x) dx`."""
    assert _nul(en_cosinus(PAIRE) - transformee_unitaire(PAIRE))


def test_le_cas_d_une_fonction_impaire_de_la_planche_12() -> None:
    """Planche 12, (3) : `-i racine(2 / pi) int de 0 a l'infini de f sin(s x) dx`."""
    impaire = X * sp.exp(-(X**2))
    assert _nul(en_sinus(impaire) - transformee_unitaire(impaire))


def test_une_fonction_qui_n_est_pas_paire_est_refusee_en_cosinus() -> None:
    """La formule en cosinus ne vaut que pour une fonction paire."""
    with pytest.raises(ValueError, match="paire"):
        en_cosinus(X * sp.exp(-(X**2)))


def test_une_fonction_qui_n_est_pas_impaire_est_refusee_en_sinus() -> None:
    """La formule en sinus ne vaut que pour une fonction impaire."""
    with pytest.raises(ValueError, match="impaire"):
        en_sinus(PAIRE)


def test_la_formule_de_convolution_de_la_planche_9() -> None:
    """Planche 9 : `F(f * g) = racine(2 pi) F(f) F(g)`, sur porte * porte."""
    porte = transformee_unitaire(PORTE)
    assert _nul(transformee_unitaire(TRIANGLE) - de_la_convolution(porte, porte))
    assert _nul(de_la_convolution(porte, porte) - RACINE * porte**2)


@pytest.mark.parametrize("fonction", [PAIRE, PORTE])
def test_l_egalite_energetique_de_la_planche_18(fonction: sp.Expr) -> None:
    """Planche 18 : `||f||^2 = ||F(f)||^2`, sans facteur."""
    spectrale = energie_spectrale(transformee_unitaire(fonction))
    assert _nul(spectrale - energie_temporelle(fonction))


def test_l_inverse_d_une_image_impaire_passe_par_le_sinus() -> None:
    """Planche 12, (4) : `racine(2 / pi) int de 0 a l'infini de b(s) sin(s x) ds`."""
    impaire = X * sp.exp(-(X**2))
    assert _nul(inverse_unitaire(transformee_unitaire(impaire)) - impaire)


def test_l_inverse_d_une_image_quelconque_passe_par_la_definition() -> None:
    """Planche 11 : une image ni paire ni impaire, partagee en ses deux parties."""
    retour = inverse_unitaire(transformee_unitaire(CAUSALE))
    assert _nul(retour.subs(X, 1) - sp.exp(-1))
    assert _nul(retour.subs(X, -1))


def test_une_integrale_que_l_on_ne_sait_pas_calculer_le_dit() -> None:
    """Regression : une `AttributeError` sans rapport sortait a la place."""
    with pytest.raises(ValueError, match="integrale"):
        integrale_sur_la_droite(sp.exp(sp.I * S * X) / (1 + S**2), S)
