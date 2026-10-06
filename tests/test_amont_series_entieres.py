"""Mathematiques II, section 11 : les exemples des planches, par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que `polytech.maths.domain.series` porte les series entieres et la
resolution d'une EDO par series tels que la section 11 les enonce, en
recopiant ses exemples.

Source : `CalculDifferentielIntegral2/SLIDES/Slides_11_Solutions_par_series_d_EDO.pdf`,
58 pages.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.series.entieres import INDICE, SerieEntiere
from polytech.maths.domain.series.equations import INCONNUES, solution_en_serie
from polytech.maths.domain.symboles import VARIABLE

N, X = INDICE, VARIABLE
A0, A1 = INCONNUES


@pytest.mark.parametrize(
    ("serie", "domaine"),
    [
        (SerieEntiere(sp.Integer(1)), sp.Interval.open(-1, 1)),
        (SerieEntiere(1 / (N + 1)), sp.Interval.Ropen(-1, 1)),
        (SerieEntiere((-1) ** N / (N + 1)), sp.Interval.Lopen(-1, 1)),
        (SerieEntiere(1 / (N * 2**N), depuis=1), sp.Interval.Ropen(-2, 2)),
    ],
    ids=["page 33, 1", "page 33, 2", "page 33, 3", "page 34"],
)
def test_le_domaine_de_convergence_des_exemples(
    serie: SerieEntiere, domaine: sp.Set
) -> None:
    """Pages 33 et 34 : le rayon, puis chaque borne testee a part.

    La serie de la page 34 est centree en 3 : `|x - 3| < 2`, et elle
    converge en `x = 1`, soit `-2` pour la variable centree.
    """
    assert serie.domaine_de_convergence() == domaine


def test_les_deux_cas_extremes_du_rayon_de_la_page_32() -> None:
    """Page 32 : `alpha = 0` donne `R` infini, `alpha` infini donne `R = 0`."""
    assert SerieEntiere(1 / sp.factorial(N)).rayon_par_le_quotient() == sp.oo
    assert SerieEntiere(sp.factorial(N)).rayon_par_le_quotient() == 0
    assert SerieEntiere(1 / (N * 2**N), depuis=1).rayon_par_la_racine() == 2


def test_la_derivee_et_la_primitive_terme_a_terme_des_pages_39_et_41() -> None:
    """Pages 39 et 41 : `somme (m + 1) a_(m+1) x^m` et `somme a_n x^(n+1) / (n + 1)`."""
    serie = SerieEntiere(1 / (N + 1))
    for rang in range(4):
        suivant = serie.coefficient(rang + 1)
        assert serie.derivee().coefficient(rang) == (rang + 1) * suivant
        attendu = serie.coefficient(rang) / (rang + 1)
        assert serie.primitive().coefficient(rang + 1) == attendu


def test_le_prototype_des_pages_45_a_49_rend_cosinus_et_sinus() -> None:
    """Pages 47 a 49 : `a_2 = -a_0 / 2`, `a_3 = -a_1 / 3!`, `a_4 = a_0 / 4!`."""
    cosinus = 1 - X**2 / 2 + X**4 / 24
    sinus = X - X**3 / 6 + X**5 / 120
    solution = solution_en_serie((1, 0, 1), 5)
    assert sp.expand(solution - A0 * cosinus - A1 * sinus) == 0


def test_l_equation_d_airy_des_pages_50_a_53() -> None:
    """Page 52 : `a_3n = a_0 / (2.3.5.6...)`, `a_(3n+1) = a_1 / (3.4.6.7...)`."""
    premiere = 1 + X**3 / (2 * 3) + X**6 / (2 * 3 * 5 * 6)
    seconde = X + X**4 / (3 * 4) + X**7 / (3 * 4 * 6 * 7)
    solution = solution_en_serie((-X, 0, 1), 7)
    assert sp.expand(solution - A0 * premiere - A1 * seconde) == 0


def test_le_procede_analytique_de_la_page_56_par_le_procede_algebrique() -> None:
    """Page 56 : pour `y'' + cos(x) y = 0`, les coefficients jusqu'a `x^6`.

    Second chemin : la planche derive l'equation en 0 ; ici le cosinus est
    remplace par son polynome de Mac Laurin et les puissances identifiees.
    """
    cosinus = 1 - X**2 / 2 + X**4 / 24 - X**6 / 720
    paire = 1 - X**2 / 2 + X**4 / 12 - X**6 / 80
    impaire = X - X**3 / 6 + X**5 / 30
    solution = solution_en_serie((cosinus, 0, 1), 6)
    assert sp.expand(solution - A0 * paire - A1 * impaire) == 0
