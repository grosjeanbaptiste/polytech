"""Les EDO du premier ordre, classees puis resolues -- dans cet ordre.

Le cours ne resout pas une EDO : il la **reconnait**, puis applique le
geste du type reconnu. Ses planches enumerent les quatre types, « EDO a
variables separees », « EDO homogenes de degre 0 », « EDO exactes », « EDO
lineaires », et chaque geste suit du type. Une bibliotheque qui appellerait
`dsolve` sauterait precisement l'etape que le corrige doit montrer.

L'ancrage vient du fascicule `Exs_02_Equations_differentielles.pdf`, dont
cette partie est lisible : il traite l'equation logistique
`y' = lambda y (1 - y/K)`, annonce ses points critiques `y = 0` et
`y = K`, et la resout « en la considerant comme une equation a variables
separees ». Les trois affirmations sont verifiees ici.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.autonomes import EquationAutonome
from polytech.maths.domain.differentielles.premier_ordre import (
    EquationDuPremierOrdre,
    Type,
)

X = sp.Symbol("x", real=True)
Y = sp.Function("y")
LAMBDA, K = sp.symbols("lambda K", positive=True)
INCONNUE = sp.Symbol("y", real=True)

LOGISTIQUE = LAMBDA * INCONNUE * (1 - INCONNUE / K)


@pytest.mark.parametrize(
    ("second_membre", "attendu"),
    [
        (INCONNUE * X, Type.VARIABLES_SEPAREES),
        (-INCONNUE / X + X**2, Type.LINEAIRE),
        ((INCONNUE**2 + X**2) / (X * INCONNUE), Type.HOMOGENE),
    ],
)
def test_chaque_type_du_cours_est_reconnu(
    second_membre: sp.Expr,
    attendu: Type,
) -> None:
    """Reconnaitre avant de resoudre : c'est l'ordre que le cours impose."""
    equation = EquationDuPremierOrdre(second_membre=second_membre)
    assert equation.type() is attendu


def test_la_logistique_est_a_variables_separees_comme_le_fascicule_le_dit() -> None:
    """« en la considerant comme une equation a variables separees »."""
    equation = EquationDuPremierOrdre(second_membre=LOGISTIQUE)
    assert equation.type() is Type.VARIABLES_SEPAREES


def test_une_equation_a_variables_separees_expose_ses_deux_membres() -> None:
    """`dy / g(y) = f(x) dx` : les deux integrales que le corrige montre."""
    equation = EquationDuPremierOrdre(second_membre=INCONNUE * X)
    en_y, en_x = equation.membres_separes()
    assert sp.simplify(en_y - 1 / INCONNUE) == 0
    assert sp.simplify(en_x - X) == 0


def test_la_solution_satisfait_l_equation() -> None:
    """Le controle intrinseque : substituer la solution et deriver."""
    equation = EquationDuPremierOrdre(second_membre=INCONNUE * X)
    assert equation.solution_verifiee()


def test_un_second_membre_sans_inconnue_reste_une_edo_du_premier_ordre() -> None:
    """`y' = f(x)` est le cas degenere : une primitive, et c'est tout."""
    equation = EquationDuPremierOrdre(second_membre=X**2)
    assert equation.type() is Type.VARIABLES_SEPAREES
    assert equation.solution_verifiee()


def test_un_type_non_reconnu_est_nomme_et_non_devine() -> None:
    """Deviner mal ferait appliquer le mauvais geste, en silence."""
    equation = EquationDuPremierOrdre(second_membre=sp.sin(X * INCONNUE) + X**X)
    with pytest.raises(ValueError, match="type non reconnu"):
        equation.type()


def test_les_points_critiques_de_la_logistique_sont_ceux_du_fascicule() -> None:
    """« les points critiques sont ceux qui annulent y' : y = 0 et y = K »."""
    autonome = EquationAutonome(second_membre=LOGISTIQUE)
    assert autonome.points_critiques() == (sp.Integer(0), K)


def test_la_stabilite_se_lit_sur_le_signe_de_la_derivee() -> None:
    """`y = 0` est instable, `y = K` est attractif : c'est l'allure esquissee."""
    autonome = EquationAutonome(second_membre=LOGISTIQUE)
    assert not autonome.est_stable(sp.Integer(0))
    assert autonome.est_stable(K)


def test_l_isocline_d_une_pente_donnee_est_une_equation_en_y() -> None:
    """Le fascicule demande l'equation des lignes isoclines et le trace de Gamma_0."""
    autonome = EquationAutonome(second_membre=LOGISTIQUE)
    isocline = autonome.isocline(sp.Integer(0))
    assert sp.simplify(isocline.lhs - isocline.rhs - LOGISTIQUE) == 0


def test_une_equation_non_autonome_est_refusee() -> None:
    """Elle n'a ni point critique ni portrait de phase : le dire plutot que rendre."""
    with pytest.raises(ValueError, match="non autonome"):
        EquationAutonome(second_membre=INCONNUE * X)
