"""Mathematiques II, section 1 : les suites numeriques, refaites par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles auxquels
`cursus/notions/CalculDifferentielIntegral2.toml` rattache un titre portent
ce que la planche enonce -- en recopiant ses exemples, page citee.

Source : `CalculDifferentielIntegral2/SLIDES/Slides_01_Suites_numeriques.pdf`.
Les pages sont celles du PDF ; la planche imprimee est entre parentheses.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from polytech.algorithmique.domain.recursivite import fibonacci
from polytech.maths.domain.point_fixe.approximations_successives import (
    approximations_successives,
)
from polytech.maths.domain.point_fixe.contraction import iterer

ITERES_DE_LA_PAGE_12 = (
    Fraction(2),
    Fraction(3, 2),
    Fraction(17, 12),
    Fraction(577, 408),
    Fraction(665857, 470832),
)
"""Page 12 (8) : le tableau des iteres, lignes 0 a 4."""

CINQUIEME_IMPRIME = Fraction(77227930, 54608393)
"""Page 12 (8), ligne 5, telle que la planche l'ecrit."""


def _moyenne_de_heron[T: (float, Fraction)](cote: T) -> T:
    """Pages 6 a 10 (6) : `x_(n+1) = (x_n^2 + 2) / (2 x_n)`."""
    return (cote * cote + 2) / (2 * cote)


def test_les_iteres_de_heron_sont_ceux_du_tableau() -> None:
    """Page 12 (8) : depuis `x_0 = 2`, les cinq premieres lignes, en fractions."""
    assert iterer(_moyenne_de_heron, Fraction(2), 4) == ITERES_DE_LA_PAGE_12


def test_le_cinquieme_itere_n_est_pas_la_fraction_imprimee() -> None:
    """Page 12 (8), ligne 5 : ecart au cours, sans effet sur les decimales.

    L'itere vaut `886731088897 / 627013566048`. La fraction imprimee est une
    autre approximation de `racine(2)`, juste aux quinze decimales affichees.
    """
    cinquieme = iterer(_moyenne_de_heron, Fraction(2), 5)[-1]
    assert cinquieme == Fraction(886731088897, 627013566048)
    assert cinquieme != CINQUIEME_IMPRIME
    assert float(CINQUIEME_IMPRIME) == pytest.approx(1.414213562373095, abs=1e-15)


def test_les_approximations_successives_convergent_vers_racine_de_deux() -> None:
    """Page 13 (9) : « la suite des iteres converge vers une valeur finie »."""
    limite, tours = approximations_successives(_moyenne_de_heron, 2.0, tolerance=1e-12)
    assert limite == pytest.approx(2**0.5, abs=1e-12)
    assert tours == len(ITERES_DE_LA_PAGE_12) + 1


def test_la_suite_de_fibonacci_est_celle_de_la_page_18() -> None:
    """Page 18 (14) : `u_0 = u_1 = 1`, puis `1, 1, 2, 3, 5, 8, 13, 21, 34, 55`."""
    attendue = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    assert [fibonacci(rang) for rang in range(10)] == attendue
