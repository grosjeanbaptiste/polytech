"""Mathematiques appliquees, 1.4.2 et 1.4.6 : la voie classique a l'ordre n.

Source : `Syllabus_Mathematiques_appliquees_2016_2017.pdf`.

- 1.4.2 : « SGENH = SGEH + SPENH » ;
- exemple 1.13 (page 28) : `y'''' - y'' = 1 + t + t^2`, dont la solution
  generale porte quatre constantes ;
- 1.4.6, exemple 1.17 (pages 30 et 31) : la superposition, pour
  `y'' + 3y' + 2y = cos t + t e^(-t) + t^2` ;
- 3.6, exemples 1 et 2 (page 59) : « par les methodes classiques », les
  constantes se tirent d'un systeme lineaire. La transformee de Laplace
  (`laplace.ordre_n`) y sert de second chemin.

Deux coquilles de l'exemple 1.17, la premiere deja consignee a la
taxonomie : le cours imprime `(cos t - 3 sin t) / 10` pour
`(cos t + 3 sin t) / 10`, et page 30 `K(D) = D^2 + 3D + 3` pour
`D^2 + 3D + 2`. Le test etablit le signe par substitution.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.comparaison import egales
from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel
from polytech.maths.domain.differentielles.ordre_n import (
    constantes,
    particuliere_par_superposition,
    solution_de_cauchy,
    solution_generale,
)
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire
from polytech.maths.domain.symboles import TEMPS as T

R = sp.Rational
K = OperateurDifferentiel(D**2 + 3 * D + 2)
"""L'operateur de l'exemple 1.17."""
CUBIQUE = OperateurDifferentiel(D**3 + 3 * D**2 + 2 * D)
"""L'operateur des exemples 1 et 2 du 3.6."""


def test_l_exemple_1_17_terme_a_terme() -> None:
    sollicitation = sp.cos(T) + T * sp.exp(-T) + T**2
    attendue = (
        (sp.cos(T) + 3 * sp.sin(T)) / 10
        + (T**2 / 2 - T) * sp.exp(-T)
        + T**2 / 2
        - R(3, 2) * T
        + R(7, 4)
    )
    obtenue = particuliere_par_superposition(K, sollicitation)
    assert egales(obtenue, attendue)
    assert egales(K.appliquer(obtenue), sollicitation)


def test_le_signe_imprime_dans_l_exemple_1_17_ne_resout_pas() -> None:
    imprimee = (sp.cos(T) - 3 * sp.sin(T)) / 10
    assert not egales(K.appliquer(imprimee), sp.cos(T))


def test_une_sollicitation_amortie_et_oscillante() -> None:
    sollicitation = sp.exp(-3 * T) * sp.sin(2 * T)
    obtenue = particuliere_par_superposition(K, sollicitation)
    assert egales(K.appliquer(obtenue), sollicitation)
    assert sp.im(obtenue) == 0


def test_une_sollicitation_qui_n_est_pas_de_la_famille_leve() -> None:
    with pytest.raises(ValueError, match="ni polynome, ni exponentielle"):
        particuliere_par_superposition(K, sp.log(T + 1))
    with pytest.raises(ValueError, match="ni polynome, ni exponentielle"):
        particuliere_par_superposition(K, sp.exp(T**2))


def test_l_exemple_1_13_a_quatre_constantes() -> None:
    operateur = OperateurDifferentiel(D**4 - D**2)
    c1, c2, c3, c4 = constantes(4)
    # Les constantes suivent l'ordre des racines : -1, puis 0 (double), puis 1.
    attendue = (
        c1 * sp.exp(-T)
        + c2
        + c3 * T
        + c4 * sp.exp(T)
        - R(3, 2) * T**2
        - T**3 / 6
        - T**4 / 12
    )
    generale = solution_generale(operateur, 1 + T + T**2)
    assert egales(operateur.appliquer(generale), 1 + T + T**2)
    assert set(generale.free_symbols) == {T, c1, c2, c3, c4}
    assert sp.expand(generale - attendue) == 0


def test_la_solution_generale_homogene_est_la_combinaison_de_la_base() -> None:
    c1, c2, c3 = constantes(3)
    assert solution_generale(CUBIQUE) == c1 * sp.exp(-2 * T) + c2 * sp.exp(-T) + c3


def test_le_3_6_exemple_1_les_constantes_par_le_systeme_lineaire() -> None:
    assert solution_de_cauchy(CUBIQUE, 0, [1, 0, 0]) == 1


def test_le_3_6_exemple_2_par_la_voie_classique_et_par_laplace() -> None:
    attendue = -R(1, 2) + sp.exp(T) / 6 + sp.exp(-T) / 2 - sp.exp(-2 * T) / 6
    classique = solution_de_cauchy(CUBIQUE, sp.exp(T), [0, 0, 0])
    assert egales(classique, attendue)
    par_laplace = ProblemeLineaire((1, 3, 2, 0), sp.exp(T), (0, 0, 0)).solution()
    assert egales(classique, par_laplace)


def test_il_faut_autant_de_conditions_que_l_ordre() -> None:
    with pytest.raises(ValueError, match="3 conditions initiales attendues, 2 recues"):
        solution_de_cauchy(CUBIQUE, 0, [1, 0])
