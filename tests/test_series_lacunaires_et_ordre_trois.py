"""Mathematiques appliquees, chapitre 4 : la limite superieure, et l'ordre trois.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`.

- Remarque 4.1 (page 78) : « il n'est pas necessaire que la limite existe
  pour pouvoir conclure », la limite superieure la remplace ;
- exemples 4.1, numeros 6 a 9 (page 79) : `cosh`, `sinh`, `cos` et `sin`,
  dont un coefficient sur deux est nul, convergent pour tout `x` ;
- commentaires du paragraphe 4.2 (page 84) : la resolution par serie vaut
  « autour d'autres valeurs `x_0` » et pour une equation non homogene,
  dont le terme independant s'ecrit en serie au point de reference.

Trois defauts de la bibliotheque y etaient : un rayon faux (1 pour le
cosinus), une erreur sans message, et l'ordre trois qui levait
`IndexError`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.series.entieres import INDICE as N
from polytech.maths.domain.series.entieres import SerieEntiere
from polytech.maths.domain.series.equations import solution_en_serie
from polytech.maths.domain.series.lacunaires import (
    rayon_par_la_limite_superieure,
)
from polytech.maths.domain.symboles import VARIABLE as X

PAIRS = 1 / sp.factorial(2 * N)
IMPAIRS = 1 / sp.factorial(2 * N + 1)
NULS = sp.Integer(0)


@pytest.mark.parametrize(
    "sous_suites",
    [
        (PAIRS, NULS),
        (NULS, IMPAIRS),
        ((-1) ** N * PAIRS, NULS),
        (NULS, (-1) ** N * IMPAIRS),
    ],
    ids=["cosh", "sinh", "cos", "sin"],
)
def test_les_exemples_6_a_9_convergent_pour_tout_x(
    sous_suites: tuple[sp.Expr, sp.Expr],
) -> None:
    assert rayon_par_la_limite_superieure(sous_suites) == sp.oo


def test_la_limite_superieure_est_la_plus_grande_des_limites() -> None:
    # a_(2m) = 2^(2m), a_(2m+1) = 3^-(2m+1) : la racine n-ieme oscille entre
    # 2 et 1/3, sans limite ; sa limite superieure vaut 2.
    sous_suites = (2 ** (2 * N), sp.Integer(3) ** (-(2 * N + 1)))
    assert rayon_par_la_limite_superieure(sous_suites) == sp.Rational(1, 2)


def test_une_seule_sous_suite_redonne_le_critere_de_la_racine() -> None:
    assert rayon_par_la_limite_superieure((1 / (N + 1),)) == 1
    assert rayon_par_la_limite_superieure((sp.factorial(N),)) == 0


def test_une_suite_toute_nulle_n_a_pas_de_rayon_a_calculer() -> None:
    with pytest.raises(ValueError, match="tous les coefficients sont nuls"):
        rayon_par_la_limite_superieure((NULS, NULS))


def test_il_faut_au_moins_une_sous_suite() -> None:
    with pytest.raises(ValueError, match="aucune sous-suite"):
        rayon_par_la_limite_superieure(())


@pytest.mark.parametrize(
    "terme",
    [sp.cos(N * sp.pi / 2) / sp.factorial(N), (1 + (-1) ** N) / (2 * sp.factorial(N))],
    ids=["cosinus", "alternance"],
)
def test_un_coefficient_sur_deux_nul_n_a_pas_de_limite_a_prendre(
    terme: sp.Expr,
) -> None:
    serie = SerieEntiere(terme)
    with pytest.raises(ValueError, match="limite superieure"):
        serie.rayon_par_la_racine()
    with pytest.raises(ValueError, match="limite superieure"):
        serie.rayon_par_le_quotient()


def test_un_seul_coefficient_nul_ne_gene_pas() -> None:
    assert SerieEntiere(N / (N + 1)).rayon_par_la_racine() == 1


def test_l_ordre_trois_a_trois_constantes_libres() -> None:
    # y''' + y = 0 : a_(n+3) = -a_n / ((n+1)(n+2)(n+3)).
    a0, a1, a2 = sp.symbols("a0 a1 a2")
    obtenue = solution_en_serie((1, 0, 0, 1), 6)
    attendue = (
        a0 * (1 - X**3 / 6 + X**6 / 720)
        + a1 * (X - X**4 / 24)
        + a2 * (X**2 - X**5 / 60)
    )
    assert sp.expand(obtenue - attendue) == 0
    residu = sp.expand(sp.diff(obtenue, X, 3) + obtenue)
    assert all(residu.coeff(X, j) == 0 for j in range(4))


def test_autour_d_un_autre_point_la_serie_est_en_x_moins_x0() -> None:
    # y' = y autour de 1 : y = a0 e^(x - 1).
    a0 = sp.Symbol("a0")
    obtenue = solution_en_serie((-1, 1), 4, autour_de=1)
    attendue = sp.series(a0 * sp.exp(X - 1), X, 1, 5).removeO()
    assert sp.expand(obtenue - attendue) == 0


def test_le_point_singulier_est_juge_au_point_de_reference() -> None:
    # (x - 1) y' + y = 0 : singulier en 1, ordinaire en 0.
    assert solution_en_serie((1, X - 1), 3) != 0
    with pytest.raises(ValueError, match="point singulier en 1"):
        solution_en_serie((1, X - 1), 3, autour_de=1)


def test_une_equation_non_homogene_developpe_son_second_membre() -> None:
    # y'' + y = e^x : y = a0 cos x + a1 sin x + (e^x - cos x - sin x) / 2.
    a0, a1 = sp.symbols("a0 a1")
    obtenue = solution_en_serie((1, 0, 1), 6, second_membre=sp.exp(X))
    exacte = a0 * sp.cos(X) + a1 * sp.sin(X) + (sp.exp(X) - sp.cos(X) - sp.sin(X)) / 2
    assert sp.expand(obtenue - sp.series(exacte, X, 0, 7).removeO()) == 0
