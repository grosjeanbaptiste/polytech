"""Le syllabus de signaux et systemes, chapitre 2 : les exercices resolus.

Source : `SignauxSystemes/SYLLABUS/Syllabus_2019_Chapitre_02_Transformees.pdf`,
pages 33 a 37. Les enonces 2.5 a 2.11 demandent des transformees de
Laplace et leurs inverses ; la page 37 en imprime les solutions, et ce
sont elles les oracles. Chacune est recoupee par un second chemin :
la transformee directe de la solution, ou les theoremes des valeurs
initiale et finale.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from polytech.maths.domain.laplace import theoremes
from polytech.maths.domain.symboles import IMAGE as P_DES_MATHS
from polytech.signaux.domain.heaviside import original
from polytech.signaux.domain.symboles import IMAGE, TEMPS
from polytech.signaux.domain.transformees import retard, transformee_de_laplace

P, T = IMAGE, TEMPS
INSTANTS = np.array([0.1, 0.7, 1.9, 4.2])


def _proches(calcule: sp.Expr, imprime: sp.Expr, tolerance: float = 1e-9) -> bool:
    ecart = sp.lambdify(T, calcule - imprime, "numpy")
    return bool(np.allclose(ecart(INSTANTS), 0, atol=tolerance))


def test_l_exercice_2_5_une_exponentielle_qui_part_de_4_et_finit_a_moins_1() -> None:
    """Graphe : 4 en zero, -1 a l'infini, constante de temps 0,8 s."""
    signal = -1 + 5 * sp.exp(-T / sp.Rational(4, 5))
    image = transformee_de_laplace(signal)
    solution = (4 * P - sp.Rational(5, 4)) / (P * (P + sp.Rational(5, 4)))
    assert sp.simplify(image - solution) == 0
    en_maths = solution.subs(P, P_DES_MATHS)
    assert theoremes.valeur_initiale(en_maths) == 4
    assert theoremes.valeur_finale(en_maths) == -1


def test_l_exercice_2_6_une_montee_puis_une_descente_retardee() -> None:
    """De 1 vers 2 en 1 s de constante de temps, puis vers 0 des `t = 8` en 2 s."""
    montee = transformee_de_laplace(2 - sp.exp(-T))
    descente = transformee_de_laplace(2 * sp.exp(-T / 2) - 2)
    solution = 2 / P - 1 / (P + 1)
    solution += sp.exp(-8 * P) * (2 / (P + sp.Rational(1, 2)) - 2 / P)
    assert sp.simplify(montee + retard(descente, 8) - solution) == 0
    apres = (2 - sp.exp(-T)) + (2 * sp.exp(-(T - 8) / 2) - 2)
    assert sp.limit(apres, T, sp.oo) == 0
    assert float(apres.subs(T, 8)) == pytest.approx(2.0, abs=1e-3)


def test_l_exercice_2_7_trois_inverses() -> None:
    assert _proches(original(1 / (P**2 + 3 * P + 2)), sp.exp(-T) - sp.exp(-2 * T))
    assert _proches(original(1 / (P**2 + 4)), sp.sin(2 * T) / 2)
    troisieme = -sp.Rational(1, 100) + T / 10 + sp.exp(-10 * T) / 100
    assert _proches(original(1 / (P**2 * (P + 10))), troisieme)


def test_l_exercice_2_8_un_retard_de_cinq_secondes() -> None:
    """`(p + 1) exp(-5 p) / (p^2 (p + 10))` : l'original de la fraction, retarde."""
    sans_retard = original((P + 1) / (P**2 * (P + 10)))
    imprime = T / 10 + sp.Rational(9, 100) - sp.Rational(9, 100) * sp.exp(-10 * T)
    assert _proches(sans_retard, imprime)
    image = retard(transformee_de_laplace(imprime), 5)
    attendue = (P + 1) * sp.exp(-5 * P) / (P**2 * (P + 10))
    assert sp.simplify(image - attendue) == 0


def test_l_exercice_2_9_a_un_numerateur_de_meme_degre() -> None:
    """`p^2 / (p^2 + 6 p + 8)` : une impulsion, puis `2 exp(-2 t) - 8 exp(-4 t)`."""
    calcule = original(P**2 / (P**2 + 6 * P + 8))
    imprime = sp.DiracDelta(T) + 2 * sp.exp(-2 * T) - 8 * sp.exp(-4 * T)
    assert sp.simplify(calcule - imprime) == 0


def test_l_exercice_2_9_b_un_pole_double_et_un_pole_simple() -> None:
    calcule = original((P + 1) / ((P + 3) ** 2 * (P + 10)))
    neuf, deux = sp.Rational(9, 49), sp.Rational(2, 7)
    imprime = neuf * sp.exp(-3 * T) - deux * T * sp.exp(-3 * T) - neuf * sp.exp(-10 * T)
    assert _proches(calcule, imprime)


def test_l_exercice_2_10_trois_poles_dont_deux_imaginaires() -> None:
    """Solution imprimee a trois chiffres : `-2,3 exp(-3 t) + 0,316 cos(t - 0,322)`."""
    image = (P**3 + P**2 + 2 * P + 1) / (P**3 + 3 * P**2 + P + 3)
    calcule = original(image)
    assert calcule.coeff(sp.DiracDelta(T)) == 1
    lisse = calcule - sp.DiracDelta(T)
    imprime = -2.3 * sp.exp(-3 * T) + 0.316 * sp.cos(T - 0.322)
    assert _proches(lisse, imprime, tolerance=2e-3)


def test_l_exercice_2_11_un_pole_a_droite_de_l_axe_imaginaire() -> None:
    """`1 / ((p^2 + 4)(p^2 - 4))` : le signal n'est pas borne."""
    image = 1 / ((P**2 + 4) * (P**2 - 4))
    hyperbolique = (sp.exp(2 * T) - sp.exp(-2 * T)) / 32
    imprime = hyperbolique + sp.cos(2 * T - 3 * sp.pi / 2) / 16
    assert _proches(original(image), imprime)
    en_maths = image.subs(P, P_DES_MATHS)
    assert theoremes.valeur_initiale(en_maths) == 0
    with pytest.raises(ValueError, match="theoreme inapplicable"):
        theoremes.valeur_finale(en_maths)
