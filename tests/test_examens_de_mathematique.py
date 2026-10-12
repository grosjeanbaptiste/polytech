"""Les examens de mathematique, refaits par la bibliotheque.

La fiche de revision s'appuie sur ces questions : ce qu'elle en dit doit
tenir. L'oracle est le corrige officiel de janvier 2025
(`Mathematique/EXAMENS/Exam_2025_Janvier_Corrige.pdf`), recopie ici avec
sa page ; le systeme de sa question 2 est tenu par
`tests/test_algebre_discussion_par_le_rang.py`.

Janvier 2022 n'a pas de corrige : ses reponses sont recoupees par un
second chemin -- la derivee d'une primitive, le report dans l'equation.
Ses formules se lisent sur l'image du PDF
(`Mathematique/EXAMENS/Exam_2022_Janvier_Enonce.pdf`), la conversion
n'en ayant rendu que le texte.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.integration.definie import IntegraleDefinie
from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.laplace.ordre_n import ProblemeLineaire
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T
from polytech.maths.domain.symboles import VARIABLE as X

# --- Janvier 2025, question 1 (page 1) ---------------------------------------

FRACTION_2025 = FractionRationnelle(4 * X**2, X**4 - 1)


def test_2025_q1_la_decomposition_est_celle_du_corrige() -> None:
    """`4x^2/(x^4 - 1) = 2/(x^2 + 1) - 1/(x + 1) + 1/(x - 1)`."""
    attendus = {2 / (X**2 + 1), -1 / (X + 1), 1 / (X - 1)}
    assert set(FRACTION_2025.termes()) == attendus


def test_2025_q1_la_primitive_est_celle_du_corrige() -> None:
    """`ln |(x - 1)/(x + 1)| + 2 arctan x`, sur `[2, 3]` ou tout est positif."""
    corrige = sp.log((X - 1) / (X + 1)) + 2 * sp.atan(X)
    ecart = sp.diff(FRACTION_2025.primitive() - corrige, X)
    assert sp.simplify(ecart) == 0


def test_2025_q1_l_integrale_vaut_les_deux_formes_du_corrige() -> None:
    """`ln(3/2) + 2 (arctan 3 - arctan 2)`, ou encore `ln(3/2) + 2 arctan(1/7)`."""
    valeur = IntegraleDefinie(4 * X**2 / (X**4 - 1), sp.Integer(2), sp.Integer(3))
    premiere = sp.log(sp.Rational(3, 2)) + 2 * (sp.atan(3) - sp.atan(2))
    seconde = sp.log(sp.Rational(3, 2)) + 2 * sp.atan(sp.Rational(1, 7))
    assert sp.simplify(valeur.valeur() - premiere) == 0
    assert abs(sp.N(valeur.valeur() - seconde, 30)) < sp.Float("1e-25")


# --- Janvier 2025, question 3 (pages 2 et 3) ---------------------------------

PROBLEME_2025 = ProblemeLineaire((1, -5, -14), -14 * T**2 - 10 * T - 26, (0, 13))


def test_2025_q3_l_image_est_celle_du_corrige() -> None:
    """`Y(p) = (13 p^3 - 26 p^2 - 10 p - 28) / (p^3 (p^2 - 5 p - 14))`."""
    corrige = (13 * P**3 - 26 * P**2 - 10 * P - 28) / (P**3 * (P**2 - 5 * P - 14))
    assert sp.simplify(PROBLEME_2025.image() - corrige) == 0


def test_2025_q3_les_fractions_simples_sont_celles_du_corrige() -> None:
    """`Y(p) = 2/p^3 + 2/p + 1/(p - 7) - 3/(p + 2)`."""
    corrige = 2 / P**3 + 2 / P + 1 / (P - 7) - 3 / (P + 2)
    assert sp.apart(PROBLEME_2025.image(), P) == corrige


def test_2025_q3_la_solution_est_celle_du_corrige() -> None:
    """`y(t) = t^2 + 2 + e^{7t} - 3 e^{-2t}` pour `t >= 0`."""
    corrige = T**2 + 2 + sp.exp(7 * T) - 3 * sp.exp(-2 * T)
    assert sp.simplify(PROBLEME_2025.solution() - corrige) == 0
    assert PROBLEME_2025.solution_verifiee()


# --- Janvier 2022, sans corrige ----------------------------------------------


def test_2022_q1_les_parties_ramenent_a_une_fraction_rationnelle() -> None:
    """`u = ln(1 + t)`, `dv = dt/t^2` : il reste `int dt / (t (1 + t))`."""
    parties = IntegrationParParties(sp.log(1 + X), 1 / X**2)
    reste = parties.integrale_restante().function
    assert sp.simplify(reste - (-1 / (X * (1 + X)))) == 0
    assert sp.simplify(sp.diff(parties.resultat(), X) - sp.log(1 + X) / X**2) == 0


def test_2022_q1_l_integrale_vaut_trois_ln_deux_moins_trois_demis_de_ln_trois() -> None:
    integrale = IntegraleDefinie(sp.log(1 + X) / X**2, sp.Integer(1), sp.Integer(2))
    attendue = 3 * sp.log(2) - sp.Rational(3, 2) * sp.log(3)
    assert sp.simplify(sp.expand_log(integrale.valeur() - attendue, force=True)) == 0
    numerique = sp.Integral(sp.log(1 + X) / X**2, (X, 1, 2)).evalf(20)
    assert abs(numerique - sp.N(attendue, 20)) < sp.Float("1e-15")


def test_2022_q3_la_solution_verifie_l_equation_et_les_deux_conditions() -> None:
    """`y'' - 4y' + 3y = x^2`, `y(0) = y'(0) = 0`."""
    probleme = ProblemeLineaire((1, -4, 3), T**2, (0, 0))
    attendue = (
        T**2 / 3 + 8 * T / 9 + sp.Rational(26, 27) + sp.exp(3 * T) / 27 - sp.exp(T)
    )
    assert sp.simplify(probleme.solution() - attendue) == 0
    assert probleme.est_solution(attendue)
