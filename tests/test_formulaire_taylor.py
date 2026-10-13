"""Le formulaire, section 4 : Taylor en un point, et Mac Laurin sans piege.

Sources :

- `Mathematique/FORMULAIRES/Formulaire_Aide_memoire.pdf`, page 5 : le
  polynome de Taylor d'ordre `k` au point `a`, son reste, et quatre
  developpements de Maclaurin ecrits jusqu'a leur terme general ;
- `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, page 143
  (imprimee 129), exercice 13 : `1/x = somme (-1)^n (x - 5)^n / 5^(n+1)` ;
- les memes notes, page 135 (imprimee 121), exemple 1 : `e^(-1/x^2)`,
  prolongee par 0, a toutes ses derivees nulles en 0 ; sa serie de
  Mac Laurin est nulle et ne lui est pas egale. La bibliotheque rendait
  `nan` pour chacun de ces coefficients.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.series.mac_laurin import coefficients_de_mac_laurin
from polytech.maths.domain.series.taylor import (
    MACLAURIN_DU_FORMULAIRE,
    coefficients_de_taylor,
    polynome_de_taylor,
    reste_de_taylor,
)
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational


def test_l_exercice_13_un_sur_x_autour_de_cinq() -> None:
    attendus = tuple((-1) ** n / sp.Integer(5) ** (n + 1) for n in range(6))
    assert coefficients_de_taylor(1 / X, 5, 6) == attendus


def test_le_polynome_est_ecrit_en_puissances_de_x_moins_a() -> None:
    polynome = polynome_de_taylor(1 / X, 5, 2)
    assert polynome == R(1, 5) - (X - 5) / 25 + (X - 5) ** 2 / 125


def test_en_zero_taylor_est_mac_laurin() -> None:
    fonction = sp.exp(X) * sp.cos(X)
    assert coefficients_de_taylor(fonction, 0, 5) == coefficients_de_mac_laurin(
        fonction, 5
    )


def test_le_reste_complete_exactement_un_polynome() -> None:
    # Pour un polynome de degre 3, le reste d'ordre 2 est exact quel que soit c.
    c = sp.Symbol("c")
    fonction = X**3 - 2 * X + 1
    reste = reste_de_taylor(fonction, 1, 2, c)
    assert sp.expand(polynome_de_taylor(fonction, 1, 2) + reste - fonction) == 0


def test_le_reste_de_l_exponentielle_porte_le_point_intermediaire() -> None:
    c = sp.Symbol("c")
    assert reste_de_taylor(sp.exp(X), 2, 3, c) == sp.exp(c) * (X - 2) ** 4 / 24


def test_un_ordre_negatif_leve() -> None:
    with pytest.raises(ValueError, match="ordre negatif"):
        polynome_de_taylor(sp.exp(X), 0, -1)


def test_une_fonction_non_definie_au_point_leve() -> None:
    with pytest.raises(ValueError, match="n'est pas derivable en 0"):
        coefficients_de_taylor(1 / X, 0, 2)


def test_le_formulaire_donne_quatre_developpements() -> None:
    assert list(MACLAURIN_DU_FORMULAIRE) == [
        sp.exp(X),
        sp.sin(X),
        sp.cos(X),
        1 / (1 + X),
    ]


@pytest.mark.parametrize("fonction", list(MACLAURIN_DU_FORMULAIRE), ids=str)
def test_chaque_terme_general_redonne_le_developpement(fonction: sp.Expr) -> None:
    terme = MACLAURIN_DU_FORMULAIRE[fonction]
    somme = sum((terme(n) for n in range(8)), sp.Integer(0))
    degre = sp.degree(somme, X)
    attendu = polynome_de_taylor(fonction, 0, degre)
    assert sp.expand(somme - attendu) == 0


def test_l_exemple_1_des_notes_a_tous_ses_coefficients_nuls() -> None:
    assert coefficients_de_mac_laurin(sp.exp(-1 / X**2), 4) == (0, 0, 0, 0)


def test_la_serie_nulle_n_est_pas_la_fonction() -> None:
    fonction = sp.exp(-1 / X**2)
    assert polynome_de_taylor(fonction, 0, 3) == 0
    assert fonction.subs(X, 1) == sp.exp(-1)


def test_un_saut_en_zero_n_a_pas_de_coefficient() -> None:
    with pytest.raises(ValueError, match="n'est pas derivable en 0"):
        coefficients_de_mac_laurin(sp.Abs(X) / X, 1)


def test_une_limite_infinie_des_deux_cotes_n_est_pas_une_valeur() -> None:
    with pytest.raises(ValueError, match="n'est pas derivable en 0"):
        coefficients_de_taylor(1 / X**2, 0, 1)
