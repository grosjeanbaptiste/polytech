"""Syllabus de mathematiques appliquees 1.3 : reconnaitre les six cas du cours.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 11 a 17 (imprimees). `EquationDuPremierOrdre.type` reconnait les
variables separees, l'homogene de degre 0 et la lineaire ; le syllabus en
ajoute trois :
- 1.3.5, l'equation **exacte** `P dx + Q dy = 0` ;
- 1.3.6 et exemple 1.4 (page 13) : `y' = -(3xy^2 + 2y) / (2x^2 y + x)`,
  « ni [...] a variables separees, ni [...] homogene. Est-ce une equation
  exacte ? [...] Non », mais `h = x` la rend exacte ;
- 1.3.8 et exemple 1.6 (page 17) : `y' + y / (x + 1) = -y^2` est « une
  equation de Bernoulli, avec a = 2 ».

`P` et `Q` sont lus sur la fraction ecrite, comme le fait l'exemple 1.4 :
la decomposition `f = -P/Q` n'est pas unique, et une autre ecriture de la
meme equation peut changer le verdict.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.bernoulli import EquationDeBernoulli
from polytech.maths.domain.differentielles.premier_ordre import (
    INCONNUE as Y,
)
from polytech.maths.domain.differentielles.premier_ordre import (
    VARIABLE as X,
)
from polytech.maths.domain.differentielles.premier_ordre import (
    EquationDuPremierOrdre,
    Type,
)
from polytech.maths.domain.differentielles.reconnaissance import (
    en_bernoulli,
    type_du_cours,
)

EXEMPLE_1_4 = -(3 * X * Y**2 + 2 * Y) / (2 * X**2 * Y + X)
EXEMPLE_1_6 = -Y / (X + 1) - Y**2


def _type(second_membre: sp.Expr) -> Type:
    return type_du_cours(EquationDuPremierOrdre(second_membre))


def test_l_exemple_1_4_demande_un_facteur_integrant() -> None:
    assert _type(EXEMPLE_1_4) is Type.FACTEUR_INTEGRANT


def test_une_forme_exacte_est_reconnue() -> None:
    """`(2xy + 1) dx + (x^2 + 2y) dy = 0` : `dP/dy = dQ/dx = 2x`."""
    assert _type(-(2 * X * Y + 1) / (X**2 + 2 * Y)) is Type.EXACTE


def test_l_exemple_1_6_est_de_bernoulli() -> None:
    assert _type(EXEMPLE_1_6) is Type.BERNOULLI


def test_l_exemple_1_6_rend_l_equation_de_bernoulli_du_cours() -> None:
    assert en_bernoulli(EXEMPLE_1_6) == EquationDeBernoulli(
        1 / (X + 1), sp.Integer(-1), sp.Integer(2)
    )


def test_un_exposant_fractionnaire_reste_de_bernoulli() -> None:
    attendue = EquationDeBernoulli(-X, X**2, sp.Rational(1, 2))
    assert en_bernoulli(X * Y + X**2 * sp.sqrt(Y)) == attendue


def test_les_trois_types_de_la_methode_passent_d_abord() -> None:
    """L'ordre du cours : separees, homogene, lineaire restent ce qu'ils etaient."""
    assert _type(Y * X) is Type.VARIABLES_SEPAREES
    assert _type(-Y / X + X**2) is Type.LINEAIRE
    assert _type((Y**2 + X**2) / (X * Y)) is Type.HOMOGENE


def test_une_equation_hors_du_cours_leve_toujours() -> None:
    with pytest.raises(ValueError, match="non reconnu"):
        _type(sp.sin(X * Y))


def test_en_bernoulli_refuse_une_equation_qui_n_en_est_pas_une() -> None:
    with pytest.raises(ValueError, match="Bernoulli"):
        en_bernoulli(sp.sin(X * Y))


def test_en_bernoulli_refuse_trois_puissances_de_y() -> None:
    with pytest.raises(ValueError, match="Bernoulli"):
        en_bernoulli(X * Y + Y**2 + X * Y**3)


def test_en_bernoulli_refuse_un_terme_sans_y() -> None:
    """`y' = x y + y^2 + x` n'a pas la forme `-P y + Q y^a`."""
    with pytest.raises(ValueError, match="Bernoulli"):
        en_bernoulli(X * Y + Y**2 + X)


def test_deux_termes_en_y_s_additionnent_dans_p() -> None:
    attendue = EquationDeBernoulli(-(X + 1), sp.Integer(1), sp.Integer(2))
    assert en_bernoulli(X * Y + Y + Y**2) == attendue


def test_un_coefficient_qui_depend_de_y_n_est_pas_de_bernoulli() -> None:
    with pytest.raises(ValueError, match="forme de Bernoulli"):
        en_bernoulli(X * Y + Y**2 * sp.sin(Y))


def test_sans_terme_en_y_ce_n_est_pas_bernoulli() -> None:
    with pytest.raises(ValueError, match="forme de Bernoulli"):
        en_bernoulli(X * Y**2)


def test_une_equation_lineaire_n_est_pas_de_bernoulli() -> None:
    """`y' = x y + x^2` : l'exposant 0 la laisse lineaire."""
    with pytest.raises(ValueError, match="forme de Bernoulli"):
        en_bernoulli(X * Y + X**2)
